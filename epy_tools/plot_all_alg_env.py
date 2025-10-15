"""
這個程式用來畫所有方法在所有環境的表現
每一個環境會有一張圖，每一個方法會有一條線
另外在 stdout 還會輸出每個方法各seeds的max的avg

輸入 -f: sacred dir
         其下一層是 method 下一層是 env
"""

import argparse
import json
import os
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file-dir', '--file_dir', type=str,
                        help='The directory containing the results', dest='file_dir', required=True)
    # Search by excluding
    parser.add_argument('-ex', '--exclude', type=str, nargs='+', help='Exclude some methods (substring)',
                        dest='exclude', required=False, default=[])
    # Filter by including
    parser.add_argument('-in', '--include', type=str, nargs='+',
                        help='Include some methods (substring); if given included strings, must have all of them',
                        dest='include', required=False, default=[])
    # Only show the result of one env
    parser.add_argument('--env', type=str, help='Only show the result of this env; '
                                                'if not given, show all envs', required=False, default=None)

    # Data Checking

    # E.g.,
    #       --cut-after 2050000 --n-test 41 (we expect ~2000000, 50000 is a period)
    #                     => after cutting, 41 in total

    parser.add_argument('--cut-after', type=int, help='Cut the test results after this number of steps',
                        required=True, dest='cut_after')
    parser.add_argument('--n-test', type=int, help='Each experiment should have n-test test results',
                        required=True, dest='n_test')
    parser.add_argument('--start-step', type=int, help='Start step', required=True, dest='start_step')
    parser.add_argument('--interval', type=int, help='Interval', required=True, dest='interval')
    parser.add_argument('--n-columns', type=int, help='Number of columns (subplots)', required=False, default=4,
                        dest='n_columns')
    parser.add_argument('--no-show', action='store_true', help='Do not show the plot', required=False, default=False,
                        dest='no_show')

    parser.add_argument('--base-width', type=int, help='Base width of the plot', required=False, default=6)
    parser.add_argument('--base-height', type=int, help='Base height of the plot', required=False, default=5)
    parser.add_argument('-yl', '--y-low', type=float, help='Y low', required=False, default=None, dest='y_low')
    parser.add_argument('-yh', '--y-high', type=float, help='Y high', required=False, default=None, dest='y_high')
    parser.add_argument('--stderr', action='store_true',
                        help='Plot stderr instead of ci (default: 95%CI)', required=False,
                        default=False, dest='stderr')

    parser.add_argument('--plot-compare-time-points', nargs='+', type=int,
                        help='Plot compare images in what time points? 注意：這裡的 time points 是從 0 開始的',
                        required=False, default=None, dest='plot_compare_time_points')

    parser.add_argument('--plot-freq', type=int, required=False, default=1, dest='plot_freq',
                        help='Plot frequency (1 for every data, 2 for every 2 data, ...)')
    parser.add_argument('--errorbar', action='store_true', required=False, default=False, dest='errorbar',
                        help='Plot errorbar instead plot + fill_between')
    parser.add_argument('--tag', '-tag', type=str, required=False, default='test_return_mean', dest='tag',
                        help='預設是畫 test_return_mean，若要畫別的，可以改這個參數。')
    parser.add_argument('--ylabel', type=str, required=False, default='Return', dest='ylabel',)
    parser.add_argument('--erase-env-str', type=str, nargs='+',required=False, default=None, dest='erase_env_str',
                        help='Erase the string in the env name')
    parser.add_argument('--ex-envs', type=str, nargs='+', required=False, default=[],
                        dest='ex_envs', help='Exclude some envs')
    parser.add_argument('--dpi', type=int, required=False, default=300, dest='dpi', )

    args = parser.parse_args()

    plt.rcParams['figure.dpi'] = args.dpi
    file_dir = args.file_dir
    band_type = '95%CI' if not args.stderr else 'stderr'

    # Get all the env & method dir
    env_to_alg_paths = defaultdict(dict)
    for alg_dir in os.listdir(file_dir):
        if not os.path.isdir(os.path.join(file_dir, alg_dir)):
            continue
        if alg_dir in args.exclude:
            print(f'{alg_dir} is excluded')
            continue
        # Check if any exclude string is in the alg_dir
        if any([exclude in alg_dir for exclude in args.exclude]):
            print(f'{alg_dir} is excluded')
            continue
        if alg_dir.startswith('_'):
            continue
        if len(args.include) > 0 and not all([include in alg_dir for include in args.include]):
            continue
        for env_dir in os.listdir(os.path.join(file_dir, alg_dir)):
            if not os.path.isdir(os.path.join(file_dir, alg_dir, env_dir)):
                continue
            alg_env_path = os.path.join(file_dir, alg_dir, env_dir)
            env_to_alg_paths[env_dir][alg_dir] = alg_env_path

    # If only show one env
    if args.env is not None:
        env_to_alg_paths = {args.env: env_to_alg_paths[args.env]}

    # Exclude some envs
    if len(args.ex_envs) > 0:
        env_to_alg_paths = {env: alg_paths for env, alg_paths in env_to_alg_paths.items() if env not in args.ex_envs}

    # Erase the string in the env name here
    # (i.e. rename the env if the string is in the env name)
    if args.erase_env_str is not None:
        for env in list(env_to_alg_paths.keys()):
            for erase_str in args.erase_env_str:
                if erase_str in env:
                    env_to_alg_paths[env.replace(erase_str, '')] = env_to_alg_paths.pop(env)


    # Sort by env names and env name length
    env_to_alg_paths = dict(sorted(env_to_alg_paths.items(), key=lambda x: x[0]))
    env_to_alg_paths = dict(sorted(env_to_alg_paths.items(), key=lambda x: len(x[0])))


    # --- For each env, subplot ---
    n_envs = len(env_to_alg_paths)
    # Decide number of rows e.g., --n-columns 3 and 4 envs => 2 rows; 3 envs => 1 row
    n_rows = n_envs // args.n_columns + (1 if n_envs % args.n_columns != 0 else 0)
    fig, axs = plt.subplots(n_rows, args.n_columns,
                            figsize=(args.n_columns * args.base_width, n_rows * args.base_height))

    # Save
    env_to_alg_all_y = defaultdict(dict)
    env_to_alg_x = defaultdict(dict)

    # Sort by env name
    env_to_alg_paths = dict(sorted(env_to_alg_paths.items(), key=lambda x: x[0]))

    # Sort by env name length
    env_to_alg_paths = dict(sorted(env_to_alg_paths.items(), key=lambda x: len(x[0])))

    # A for loop over all envs
    plt.style.use('ggplot')
    for i, (env, alg_paths) in enumerate(env_to_alg_paths.items()):
        # Get the subplot
        if n_rows == 1:
            if args.n_columns == 1:
                ax = axs
            else:
                ax = axs[i % args.n_columns]
        else:
            ax = axs[i // args.n_columns, i % args.n_columns]

        # Get all the results
        all_alg_y = defaultdict(list)
        all_alg_ci = defaultdict(list)
        all_alg_avg_max = defaultdict(float)
        all_alg_avg_width_for_max = defaultdict()
        all_alg_avg_last = defaultdict(float)
        all_alg_avg_width_for_last = defaultdict()
        all_alg_x = defaultdict(list)
        all_alg_n_samples = defaultdict(int)
        for alg, alg_path in alg_paths.items():
            all_y = []
            for exp_dir in os.listdir(alg_path):
                exp_dir_path = os.path.join(alg_path, exp_dir)
                if exp_dir.startswith('_'):
                    continue
                if not os.path.isdir(exp_dir_path):
                    continue
                files = os.listdir(exp_dir_path)
                assert 'metrics.json' in files, f'{exp_dir} does not have metrics.json'
                assert 'config.json' in files, f'{exp_dir} does not have config.json'
                metric_path = os.path.join(exp_dir_path, 'metrics.json')
                config_path = os.path.join(exp_dir_path, 'config.json')
                metric_data = json.load(open(metric_path, 'r'))
                config_data = json.load(open(config_path, 'r'))

                try:
                    test_return_step_to_mean = {step: mean for step, mean in
                                                zip(metric_data[args.tag]['steps'],
                                                    metric_data[args.tag]['values'])}
                except KeyError:
                    print(f'{exp_dir} does not have test_return_mean')
                    continue

                # Cut
                for step in list(test_return_step_to_mean.keys()):
                    if step > args.cut_after:
                        test_return_step_to_mean.pop(step)

                # assert len(test_return_step_to_mean) == args.n_test, (
                #     f'{exp_dir} does not have {args.n_test} test results; '
                #     f'got {len(test_return_step_to_mean)}\n'
                #     f'; its steps: {test_return_step_to_mean.keys()}')

                if len(test_return_step_to_mean) != args.n_test:
                    warnings.warn(f'{exp_dir} does not have {args.n_test} test results; '
                                  f'got {len(test_return_step_to_mean)}\n'
                                  f'; its steps: {test_return_step_to_mean.keys()}')

                all_y.append(list(test_return_step_to_mean.values()))
                all_alg_x[alg] = list(test_return_step_to_mean.keys())

            # If not all y have the same length, we need to cut them
            min_len = min([len(y) for y in all_y])
            all_y = [y[:min_len] for y in all_y]

            # Save
            env_to_alg_all_y[env][alg] = all_y

            # Record number of samples
            all_alg_n_samples[alg] = len(all_y)

            # Compute avg and ci
            y_mean_over_seeds = np.mean(np.array(all_y), axis=0)

            # if only_one_seed set sem = 0 array
            if len(all_y) == 1:
                sem = np.zeros_like(y_mean_over_seeds)
                # sem = stats.sem(np.array(all_y), axis=0)
            else:
                sem = stats.sem(np.array(all_y), axis=0)
            all_alg_y[alg] = y_mean_over_seeds
            if args.stderr:
                # 上下界
                all_alg_ci[alg] = (y_mean_over_seeds - sem, y_mean_over_seeds + sem)
            else:
                ci = stats.t.interval(0.95, len(all_y) - 1, loc=y_mean_over_seeds, scale=sem)
                all_alg_ci[alg] = ci

            # Compute the mean of all seeds' max
            all_seed_max = [max(y) for y in all_y]
            avg_max = np.mean(all_seed_max)
            all_alg_avg_max[alg] = avg_max
            # ci
            sem = stats.sem(all_seed_max)
            if args.stderr:
                # 上下界
                all_alg_avg_width_for_max[alg] = (avg_max - sem, avg_max + sem)
            else:
                ci = stats.t.interval(0.95, len(all_seed_max) - 1, loc=avg_max, scale=sem)
                # When all numbers are the same, the ci will be nan
                # In this case, we set the ci to 0
                if sem == 0:
                    ci = (avg_max, avg_max)
                all_alg_avg_width_for_max[alg] = ci

            # Compute the mean of all seeds' last
            all_seed_last = [y[-1] for y in all_y]
            avg_last = np.mean(all_seed_last)
            all_alg_avg_last[alg] = avg_last
            # ci
            sem = stats.sem(all_seed_last)
            if args.stderr:
                # 上下界
                all_alg_avg_width_for_last[alg] = (avg_last - sem, avg_last + sem)
            else:
                ci = stats.t.interval(0.95, len(all_seed_last) - 1, loc=avg_last, scale=sem)
                # When all numbers are the same, the ci will be nan
                # In this case, we set the ci to 0
                if sem == 0:
                    ci = (avg_last, avg_last)
                all_alg_avg_width_for_last[alg] = ci

        # Sort alg by names
        all_alg_y = dict(sorted(all_alg_y.items(), key=lambda x: x[0]))
        all_alg_x = dict(sorted(all_alg_x.items(), key=lambda x: x[0]))
        # Sort by name len
        all_alg_y = dict(sorted(all_alg_y.items(), key=lambda x: len(x[0])))
        all_alg_x = dict(sorted(all_alg_x.items(), key=lambda x: len(x[0])))
        #
        # Plot
        for i_alg, alg in enumerate(all_alg_y.keys()):
            y = all_alg_y[alg]
            ci = all_alg_ci[alg]
            alg_revised_name = alg
            if len(args.include) > 0:
                for include in args.include:
                    alg_revised_name = alg_revised_name.replace(include, '')
            x = range(args.start_step, args.cut_after + 1, args.interval)
            # x = all_alg_x[alg] # TODO: 每個 run 的 x 會不完全一樣，所以無法用這個
            # 真正在比的是"第n次"的結果，而非第?個 step 的結果
            # 因為是在 episode 和 episode 之間記錄的資訊，若 episode 長度不一樣，則 step 也不一樣

            # Cut x to the same length as y
            x = x[:len(y)]
            # Save
            env_to_alg_x[env][alg] = x
            #
            label = (f'{alg_revised_name:12} ('
                     f'n={all_alg_n_samples[alg]},'
                     f'max: {all_alg_avg_max[alg]:.2f}±{all_alg_avg_width_for_max[alg][1] - all_alg_avg_max[alg]:.2f},'
                     f'last: {all_alg_avg_last[alg]:.2f}±'
                     f'{all_alg_avg_width_for_last[alg][1] - all_alg_avg_last[alg]:.2f}'
                     f')')
            #
            plotted_x = list(x)[::args.plot_freq]
            plotted_y = y[::args.plot_freq]
            ci = [ci[0][::args.plot_freq], ci[1][::args.plot_freq]]
            if args.errorbar:
                ax.errorbar(plotted_x, plotted_y, yerr=ci[1] - plotted_y, label=label, linewidth=2)
            else:

                # # Create a custom colormap transitioning from blue to yellow to red
                # cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["blue", "gray", "red"])
                import matplotlib.colors as mcolors

                custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["orange", "red"])
                #
                # # Normalize colors using the custom colormap
                # colors = cmap(np.linspace(0, 1, 11))
                cool_cmap = plt.cm.cool
                autumn_cmap = plt.cm.autumn

                # cool_colors = cool_cmap(np.linspace(0, 1, len(all_alg_y)))
                cool_colors = plt.cm.Blues(np.linspace(0.2, 0.7, len(all_alg_y)))
                cool_colors = plt.cm.Blues(np.linspace(0.2, 1, len(all_alg_y)))
                # autumn_colors = autumn_cmap(np.linspace(0., 0.5, len(all_alg_y)))
                autumn_colors = custom_cmap(np.linspace(0., 1, len(all_alg_y)))


                color = cool_colors[i_alg]


                color = None
                linewidth = 1.5

                if '@' in alg or 'equal' in alg or 'arithmetic' in alg:

                    linestyle = '--'
                    linestyle = 'dotted'
                    # 'densely dashed'
                    linestyle = (0, (1, 1))

                    if 'largerBuffer' in alg:
                        linestyle = '-'


                    color = autumn_colors[i_alg]
                    if 'equal' in alg:
                        if 'gradNext' in alg:
                            color = 'green'
                        else:
                            color = 'orange'
                    elif 'arithmetic' in alg:
                        color = 'red'
                    elif '@' in alg:
                        color = 'green'
                    linewidth = 1
                    # color=None
                else:
                    if '_15s' in alg:
                        # pass
                        color = 'black'
                        linewidth = 3
                    else:
                        color = cool_colors[i_alg]

                    # if '1s' in alg:

                    # For SE env
                    if 'ssm' in alg:
                        color = 'red'
                        plotted_x = (np.array(plotted_x) / 2).tolist()
                        linewidth = 2
                    elif 'esm' in alg:
                        color = 'darkviolet'
                        # plotted_x = (np.array(plotted_x) / 2).tolist()
                        linewidth = 2
                    elif 'iql_w10000_' == alg:
                        color = 'red'
                    elif 'iql_w1000_' == alg:
                        color = 'blue'
                    elif 'iql_w2000_' == alg:
                        color = 'green'
                    # elif 'iql_w10000_addSightId' == alg:
                    #     color = 'C2'

                    linestyle = '-'

                    # if '1s' in alg:
                    #     color = 'blue'
                    # elif '2s' in alg:
                    #     color = 'green'
                    # elif '3s' in alg:
                    #     color = 'red'
                    # if 'buffer100' in alg:
                    #     linestyle = '--'

                    # else:
                    #     color = cool_colors[i_alg]

                print(f'alg: {alg}  ')
                ax.plot(plotted_x, plotted_y, label=label, linewidth=linewidth, linestyle=linestyle, color=color)
                ax.fill_between(plotted_x, ci[0], ci[1], alpha=0.1, color=color)

                # If alg str too long, we need to cut it and add ...
                if len(alg) > 12:
                    alg_for_text = alg[:12] + '...'
                else:
                    alg_for_text = alg

                # Text: at the end of the line
                ax.text(plotted_x[-1], plotted_y[-1], alg_for_text, fontsize=9, color='black', verticalalignment='center')
                # Text: at the 2/10
                plot_text_positions = [2, 5]
                plot_text_positions = []


                for iii in plot_text_positions:
                    ax.text(plotted_x[int(len(plotted_x) / 10 * iii)], plotted_y[int(len(plotted_y) / 10 * iii)],
                            alg_for_text, fontsize=6, color='black', verticalalignment='center')

                # # Original
                # ax.plot(plotted_x, plotted_y, label=label, linewidth=2)
                # ax.fill_between(plotted_x, ci[0], ci[1], alpha=0.1)

        env_for_title = env
        title_str = f'{env_for_title} \n(tag={args.tag}) (mean±{band_type})'
        if len(args.include) > 0:
            title_str += f' ({" ".join(args.include)})'
        ax.set_title(title_str)

        ax.legend()
        ax.grid()
        ax.set_xlabel('Steps')
        ax.set_ylabel(args.ylabel)
        # ax.set_ylim(0, 1.0)
        # ax.set_ylim(0, 1.0)
        # if args.y_low is not None:
        #     ax.set_ylim(args.y_low, None)
        ax.set_ylim(args.y_low, args.y_high)
    # plt.tight_layout()
    plt.savefig('tmp.png')
    print(f'Saved to tmp.png')
    if not args.no_show:
        plt.show()
    plt.close()

    # Plot comparisons at different time points
    if isinstance(args.plot_compare_time_points, list) and len(args.plot_compare_time_points) > 0:
        for env, alg_all_y in env_to_alg_all_y.items():
            for time_point in args.plot_compare_time_points:
                plt.figure(figsize=(15, 8))
                steps = []
                # sort by name and len
                alg_all_y = dict(sorted(alg_all_y.items(), key=lambda x: x[0]))
                alg_all_y = dict(sorted(alg_all_y.items(), key=lambda x: len(x[0])))
                #
                for i_alg, (alg, all_y) in enumerate(alg_all_y.items()):
                    steps.append(env_to_alg_x[env][alg][time_point])
                    y = [y[time_point] for y in all_y]
                    c = 'red'
                    plt.boxplot(y, positions=[i_alg], widths=0.6,
                                boxprops=dict(color=c) if "@" in alg else None,
                                capprops=dict(color=c) if "@" in alg else None,
                                whiskerprops=dict(color=c) if "@" in alg else None,
                                )
                assert all([step == steps[0] for step in steps]), 'Not all x have the same values'
                plt.title(f'{env}: steps={steps[0]} | tag = {args.tag} | time point = {time_point}')
                plt.xlabel('Algorithms')
                plt.ylabel(args.ylabel)

                # Split into lines if alg names are too long
                # 每 12 個字元換行
                xticks_alg_names = []
                for alg_name in list(alg_all_y.keys()):
                    new_str = ""
                    for i in range(0, len(alg_name), 12):
                        new_str += alg_name[i:i + 12] + '\n'
                    xticks_alg_names.append(new_str)

                plt.xticks(range(len(alg_all_y)), xticks_alg_names, rotation=45)
                plt.yticks(np.arange(0, 0.6, 0.1))
                plt.tight_layout()
                plt.grid()
                plt.savefig(f'{env}_time_point_{time_point}.png')
                if not args.no_show:
                    plt.show()
                plt.close()
