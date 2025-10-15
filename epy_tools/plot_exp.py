"""給定 config yaml，畫出圖片
使用和 `check_exp.py` 所用一樣的 config yaml
"""

import argparse
import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from sacred.optional import yaml
from scipy import stats
from tqdm import tqdm

# Constants
BAND_NAMES = {'ci': '95%CI', 'stderr': 'stderr', 'minmax': 'min-max', 'std': 'std'}

# Global config (from yaml)
CONFIG: Optional[dict] = None

IMG_SAVE_DIR = 'exp_plot'


@dataclass
class PlotConfig:  # "pc": abbreviation of `PlotConfig`
    # The default is an example for plotting test_return_mean

    # Save path (if None or empty, do not save img) (if 'auto', it will automatically produce a name)
    save_path: str
    # ---------- Data ----------
    # Path to the sacred dir. Use argparse to get this if not given in the yaml
    # Recommendation: not given in the yaml
    f: Optional[str] = None
    # Title
    title: Optional[str] = None
    # Env (map names)
    maps: list[str] = field(default_factory=list)
    # Alg
    algorithms: list[str] = field(default_factory=list)
    # Sight settings
    sights: list[str] = field(default_factory=list)
    # Tag to plot
    tag: str = 'test_return_mean'
    # Number of data to plot; if None, get min length of all runs
    # If 'auto':
    #            if tag == 'test_return_mean', use the `n_min_tests` specified in the config yaml
    #            if other tags, use the min length of all runs
    n_tag_data: Optional[Union[int, str]] = 'auto'
    # Available: 'ci' (95%ci), 'std', 'stderr', 'minmax' (point is mean) # TODO
    band: str = 'ci'
    # Plot frequency (1 for every data, 2 for every 2 data, ...) (first data is idx=0)
    plot_skip: int = 1
    # ---------- Display ----------
    # Each graph's size
    base_width: int = 12
    base_height: int = 9
    # Each axis's limit (if None, use the default of matplotlib)
    y_low: Optional[float] = None
    y_high: Optional[float] = None
    x_low: Optional[float] = None
    x_high: Optional[float] = None
    # Labels
    xlabel: str = 'Steps'
    ylabel: str = 'Return'
    # 符合條件的 curve 會被標記成給定的顏色(例如可以以此來標記 original sight)
    # `list_of_assigned_color_and_keywords` 是一個 list
    # 每要一種 plot 自訂化風格，放一組 dict 在 list。
    # 這個 dict 要包含至少一個 key: "keywords"，其 value 是一個 list of str，代表要符合的 keywords
    # 另外，根據需要，要改 color / linestyle / linewidth / ... (plot的kwargs) 等等就加上對應的 key 和 value，
    #               若符合keywords則會套用
    # 屆時檢查時，會把資料的alg env sight放到set，然後一個一個檢查 keywords 是否全部在裡面，有則符合
    list_of_assigned_styles_and_keywords: Optional[dict] = field(default_factory=list)
    # 標註有哪些 pure sight (也可以不含)。這是用來輔助pure sight 深淺顏色的設定
    pure_sights: Optional[list[str]] = None
    # Line width
    linewidth: int = 2
    # DPI: higher DPI means higher resolution
    dpi: int = 300
    # Whether to show the plot (plt.show())
    no_show: bool = False
    #
    label_without_overall_performance: bool = False
    hide_title_description: bool = False
    show_title: bool = True
    show_legend: bool = True
    show_text_on_line: bool = True


def get_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, help='Sacred dir path', dest='f', required=True)
    parser.add_argument('-c', type=str, help='Config yaml path', dest='config', required=True)

    args = parser.parse_args()

    # Load yaml config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load plot config. plot_config (assume non-empty) is a list of each plot's config
    assert config.get('plot', None) is not None, 'No plot config found in the config yaml'
    pcs: list[PlotConfig] = []
    for candidate in config['plot']['candidates']:
        if candidate['save_path'] not in config['plot']['run']:  # Here use save_path to distinguish
            continue
        pc = PlotConfig(save_path=candidate['save_path'])
        pc.f = args.f
        for k, v in candidate.items():  # 用 config yaml 給的來覆蓋預設值
            setattr(pc, k, v)
        pcs.append(pc)

    # Function run
    run_dict: dict
    for run_dict in config['plot']['function_run']:
        assert 'template_save_path' in run_dict, 'No template_save_path in function_run'

        template_save_path = run_dict.pop('template_save_path')
        pc = PlotConfig(save_path=template_save_path)
        pc.f = args.f

        # Frm a list of dicts, find the dict whose save_path equals to template_save_path
        candidate: Optional[dict] = next(
            (c for c in config['plot']['candidates'] if c['save_path'] == template_save_path), None)

        if candidate is None:
            raise ValueError(f'No candidate found for template_save_path: {template_save_path}')

        for k, v in candidate.items():  # 用 config yaml 給的來覆蓋預設值
            setattr(pc, k, v)
        for k, v in run_dict.items():  # 用 function run yaml 給的來覆蓋 template 預設值
            setattr(pc, k, v)

        pcs.append(pc)

    global CONFIG
    CONFIG = config
    return pcs


def get_env_id_by_map_and_sight(map_name: str, sight: str) -> str:
    return CONFIG['map_to_sight_to_sight_and_env_id'][map_name][sight]['env_id']


def plot_one(pc: PlotConfig):
    # This graph's settings
    plt.rcParams['figure.dpi'] = pc.dpi
    plt.style.use('ggplot')

    # Get all the env & method dir (since assuming the `check_exp.py` is done, we can directly derive the paths
    # using the `f` and `config` yaml)
    map_to_alg_to_sight_to_path: dict[str, dict[str, dict[str, str]]] = {}
    for map_name in pc.maps:
        alg_to_sight_paths = {}
        for alg in pc.algorithms:
            sight_to_paths = {}
            for sight in pc.sights:  # sight here is str. Generally describing the sight setting (e.g., 3s, ucb...)
                env_id = get_env_id_by_map_and_sight(map_name, sight)
                path = os.path.join(pc.f, f'{alg}_{sight}', env_id)
                # Check if the path exists
                if not os.path.exists(path):
                    warnings.warn(f'{path} does not exist')
                    continue
                sight_to_paths[str(sight)] = path
            alg_to_sight_paths[alg] = sight_to_paths
        map_to_alg_to_sight_to_path[map_name] = alg_to_sight_paths

    # Subplot ---
    n_maps = len(map_to_alg_to_sight_to_path)
    n_alg = len(pc.algorithms)
    n_rows, n_cols = n_alg, n_maps
    print(f'Plotting {n_maps} maps, {n_alg} algs, {len(pc.sights)} sights')
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * pc.base_width, n_rows * pc.base_height))

    # Reference: https://stackoverflow.com/questions/25812255/row-and-column-headers-in-matplotlibs-subplots
    row_title_pad = 65  # in points
    col_title_pad = 15  # in points
    col_titles = list(map_to_alg_to_sight_to_path.keys())
    row_titles = list(map(lambda x: x.strip().upper(), pc.algorithms))
    if len(col_titles) > 0 and len(row_titles) > 0:
        axs_for_titles = axs
    elif len(col_titles) > 0 and len(row_titles) == 0:
        axs_for_titles = axs.reshape(1, -1)
    elif len(col_titles) == 0 and len(row_titles) > 0:
        axs_for_titles = axs.reshape(-1, 1)
    else: # Only one row and one col
        axs_for_titles = axs.reshape(1, 1)
    for ax, col in zip(axs_for_titles[0,:], col_titles):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, col_title_pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='xx-large', ha='center', va='baseline')
    for ax, row in zip(axs_for_titles[:,0], row_titles):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - row_title_pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='xx-large', ha='left', va='center')

    # Starting the loop to plot each grid in the subplot
    p_bar = tqdm(total=n_maps * n_alg * len(pc.sights))
    for i_map, (map_name, alg_paths) in enumerate(map_to_alg_to_sight_to_path.items()):

        for i_alg, alg in enumerate(alg_paths.keys()):

            # Get the subplot (rows: alg, cols: map_name)
            if n_maps == 1 and n_alg == 1:
                ax = axs
            elif n_maps == 1:
                ax = axs[i_alg]
            elif n_alg == 1:
                ax = axs[i_map]
            else:
                ax = axs[i_alg, i_map]

            # if n_rows == 1:
            #     if n_cols == 1:
            #         ax = axs
            #     else:
            #         ax = axs[i_map % n_cols]
            # else:
            #     ax = axs[i_map // n_cols, i_map % n_cols]

            # y_highs, y_lows, x_highs, x_lows = [], [], [], []
            for i_sight, sight in enumerate(alg_paths[alg].keys()):

                runs_path = alg_paths[alg][sight]
                sight_x = []
                sight_y = []

                for exp_dir in os.listdir(runs_path):
                    exp_dir_path = os.path.join(runs_path, exp_dir)
                    if exp_dir.startswith('_') or not os.path.isdir(exp_dir_path):
                        continue  # Skip the non-dir files and the hidden files

                    assert 'metrics.json' in os.listdir(exp_dir_path), f'{exp_dir} does not have metrics.json'
                    metric_path = os.path.join(exp_dir_path, 'metrics.json')
                    metric_data = json.load(open(metric_path, 'r'))

                    try:
                        test_return_step_to_mean = {step: mean for step, mean in
                                                    zip(metric_data[pc.tag]['steps'],
                                                        metric_data[pc.tag]['values'])}
                    except KeyError:
                        warnings.warn(f'{metric_path} does not have test_return_mean')
                        continue

                    sight_x.append(list(test_return_step_to_mean.keys()))
                    sight_y.append(list(test_return_step_to_mean.values()))

                assert len(sight_x) > 0, f'No data found in {runs_path}'

                # Set the number of data limit
                if isinstance(pc.n_tag_data, str):
                    if pc.n_tag_data == 'auto':
                        if pc.tag == 'test_return_mean':
                            n_tag_data = CONFIG['possible_algorithms'][alg]['min_n_tests']
                        else:
                            n_tag_data = min([len(y) for y in sight_y])
                    else:
                        raise ValueError(f'Invalid n_tag_data: {pc.n_tag_data}')
                else:
                    n_tag_data = pc.n_tag_data

                # Assume all run have >= n_tag_data
                if not all([len(y) >= n_tag_data for y in sight_y]):
                    min_len = min([len(y) for y in sight_y])
                    warnings.warn(f'Not all y have >= n_tag_data. Cut them to {min_len}')
                    sight_y = [y[:min_len] for y in sight_y]
                    sight_x = [x[:min_len] for x in sight_x]

                # assert all([len(y) >= n_tag_data for y in sight_y]), 'Not all y have >= n_tag_data'
                # TODO

                # Cut them to `n_tag_data`
                sight_x = np.array([x[:n_tag_data] for x in sight_x])
                sight_y = np.array([y[:n_tag_data] for y in sight_y])
                sight_n_runs = len(sight_y)

                # Compute avg and ci
                sight_y_mean = np.mean(np.array(sight_y), axis=0)
                sight_x_mean = np.mean(np.array(sight_x), axis=0)

                # Compute the mean of all seeds' max
                sight_each_seed_max = [max(y) for y in sight_y]
                sight_each_seed_max_mean = np.mean(sight_each_seed_max)

                # Compute mean of all seeds' last
                sight_each_seed_last = [y[-1] for y in sight_y]
                sight_each_seed_last_mean = np.mean(sight_each_seed_last)

                # Compute 整體性
                # If only one run -> bandwidth is set to 0 manually
                # Init using mean
                if sight_n_runs <= 1:
                    sight_y_upp = sight_y_mean
                    sight_y_low = sight_y_mean
                    sight_each_seed_max_upp = sight_each_seed_max_mean
                    sight_each_seed_max_low = sight_each_seed_max_mean
                    sight_each_seed_last_upp = sight_each_seed_last_mean
                    sight_each_seed_last_low = sight_each_seed_last_mean
                else:
                    if pc.band == 'std':
                        sight_y_upp = sight_y_mean + np.std(sight_y, axis=0)
                        sight_y_low = sight_y_mean - np.std(sight_y, axis=0)
                        sight_each_seed_max_upp = sight_each_seed_max_mean + np.std(sight_each_seed_max)
                        sight_each_seed_last_upp = sight_each_seed_last_mean + np.std(sight_each_seed_last)
                    elif pc.band == 'stderr':
                        sem = stats.sem(sight_y, axis=0)
                        sight_y_upp = sight_y_mean + sem
                        sight_y_low = sight_y_mean - sem
                        sight_each_seed_max_upp = sight_each_seed_max_mean + stats.sem(sight_each_seed_max)
                        sight_each_seed_last_upp = sight_each_seed_last_mean + stats.sem(sight_each_seed_last)
                    elif pc.band == 'ci':
                        (sight_y_low, sight_y_upp) = stats.t.interval(0.95, len(sight_y) - 1, loc=sight_y_mean,
                                                                      scale=stats.sem(sight_y))
                        (sight_each_seed_max_low, sight_each_seed_max_upp) = stats.t.interval(0.95, len(
                            sight_each_seed_max) - 1, loc=sight_each_seed_max_mean, scale=stats.sem(
                            sight_each_seed_max))
                        (sight_each_seed_last_low, sight_each_seed_last_upp) = stats.t.interval(0.95, len(
                            sight_each_seed_last) - 1, loc=sight_each_seed_last_mean, scale=stats.sem(
                            sight_each_seed_last))
                    elif pc.band == 'minmax':
                        sight_y_upp = np.max(sight_y, axis=0)
                        sight_y_low = np.min(sight_y, axis=0)
                        sight_each_seed_max_upp = np.max(sight_each_seed_max)
                        sight_each_seed_max_low = np.min(sight_each_seed_max)
                        sight_each_seed_last_upp = np.max(sight_each_seed_last)
                        sight_each_seed_last_low = np.min(sight_each_seed_last)
                    else:
                        raise ValueError(f'Invalid band: {pc.band}')

                # =============== Plot a line ===============

                alg_and_sight = f'{alg} + {sight}'

                # Frequency of plotting
                sight_x_plotted = sight_x_mean[::pc.plot_skip]
                sight_y_plotted = sight_y_mean[::pc.plot_skip]
                sight_band_low = sight_y_low[::pc.plot_skip]
                sight_band_upp = sight_y_upp[::pc.plot_skip]

                # Skip if x_low is set
                if pc.x_low is not None:
                    # Get the index that start from x_low
                    idx = np.argmax(sight_x_plotted >= pc.x_low)  # It selects the first index that >= pc.x_low
                    sight_x_plotted = sight_x_plotted[idx:]
                    sight_y_plotted = sight_y_plotted[idx:]
                    sight_band_low = sight_band_low[idx:]
                    sight_band_upp = sight_band_upp[idx:]

                # Label contains alg, sight and the result of the sight
                label = f'{alg_and_sight:12}'

                if not pc.label_without_overall_performance:
                    label += (f' (len:{len(sight_x_plotted)},runs:{sight_n_runs},'
                              f'max:{sight_each_seed_max_mean:.2f}'
                              f'±{sight_each_seed_max_upp - sight_each_seed_max_mean:.2f},'
                              f'last:{sight_each_seed_last_mean:.2f}'
                              f'±{sight_each_seed_last_upp - sight_each_seed_last_mean:.2f})')

                # Plot kwargs
                plot_kwargs = dict(linewidth=pc.linewidth)

                # Default colors (淺到深)
                # # Create a custom colormap transitioning from blue to yellow to red
                # cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["blue", "gray", "red"])
                # # Normalize colors using the custom colormap
                # cool_cmap = plt.cm.cool
                cool_colors = plt.cm.Blues(
                    np.linspace(0.2, 1, len(pc.pure_sights) if pc.pure_sights else len(pc.sights)))
                # cool_colors = plt.cm.Blues(np.linspace(0.2, 1, len(pc.sights)))

                # Get the idx of this side in the pc.sights if given
                color_idx = None
                if pc.pure_sights and sight in pc.pure_sights:
                    color_idx = pc.pure_sights.index(sight)
                plot_kwargs['color'] = cool_colors[color_idx] if color_idx is not None else None

                # Check custom style
                n_satisfied = 0
                for style_dict in pc.list_of_assigned_styles_and_keywords:
                    keywords = style_dict['keywords']
                    # 2024025 version: only match substring ✓
                    if all([(keyword in {alg, map_name, sight}) or (keyword in alg) or (keyword in map_name) or (keyword in sight) for keyword in
                            keywords]):  # 標註有哪些 pure sight (也可以不含)。這是用來輔助pure sight 深淺顏色的設定

                        # If the keywords are all in the alg map_name sight, apply the kwargs
                        plot_kwargs.update({k: v for k, v in style_dict.items() if k != 'keywords'})
                        n_satisfied += 1
                # assert n_satisfied <= 1, (f'One sight can only have one style, but {n_satisfied} styles are satisfied. '
                #                           f'Please check the config yaml')
                # 20240825 version: multiple styles can be satisfied 不過後會覆蓋前

                # # 簡單看，sight名非 (2字且最後是s) 也非 (3字且最後是s) 基本不是 pure sight
                # if not (len(sight) == 2 and sight[-1] == 's') and not (len(sight) == 3 and sight[-1] == 's'):
                #     color = 'red'

                # Plot
                ax.plot(sight_x_plotted, sight_y_plotted, label=label, **plot_kwargs)

                # Fill between
                fill_between_kwargs = dict(alpha=0.1)
                if 'color' in plot_kwargs:
                    fill_between_kwargs['color'] = plot_kwargs['color']
                ax.fill_between(sight_x_plotted, sight_band_low, sight_band_upp, **fill_between_kwargs)

                # If alg str too long, we need to cut it and add ...
                if len(alg_and_sight) > 12:
                    alg_and_sight_for_text = alg_and_sight[:12] + '...'
                else:
                    alg_and_sight_for_text = alg_and_sight

                # Text: at the end of the line and other positions
                if pc.show_text_on_line:
                    ax.text(sight_x_plotted[-1], sight_y_plotted[-1], alg_and_sight_for_text, fontsize=9, color='black',
                            verticalalignment='center')
                    plot_text_positions = []  # e.g., [2, 5] (十分之幾)
                    for plot_text_position in plot_text_positions:
                        ax.text(sight_x_plotted[int(len(sight_x_plotted) / 10 * plot_text_position)],
                                sight_y_plotted[int(len(sight_y_plotted) / 10 * plot_text_position)],
                                alg_and_sight_for_text, fontsize=6, color='black', verticalalignment='center')

                if pc.show_title:
                    title_desc = '\n(tag={pc.tag}) (mean±{BAND_NAMES[pc.band]})' if not pc.hide_title_description else ''
                    title_str = f'{map_name}{title_desc}'
                    ax.set_title(title_str)
                if pc.show_legend:
                    ax.legend()
                ax.grid()
                ax.set_xlabel('Steps')
                ax.set_ylabel(pc.ylabel)
                p_bar.update(1)

            #     # If limit is None, compute the min/max of the data to set the limit
            #     # x_low = min(sight_x_plotted) - np.abs(min(sight_x_plotted)) * .0 if pc.x_low is None else pc.x_low
            #     # x_high = max(sight_x_plotted) + np.abs(max(sight_x_plotted)) * .0 if pc.x_high is None else pc.x_high
            #     # y_low = min(sight_y_plotted) - np.abs(min(sight_y_plotted)) * .0 if pc.y_low is None else pc.y_low
            #     # y_high = max(sight_y_plotted) + np.abs(max(sight_y_plotted)) * .0 if pc.y_high is None else pc.y_high
            #     y_low = pc.y_low
            #     y_high = pc.y_high
            #     x_low = pc.x_low
            #     x_high = pc.x_high
            #     x_lows.append(x_low)
            #     x_highs.append(x_high)
            #     y_lows.append(y_low)
            #     y_highs.append(y_high)
            #
            #
            # if len(x_lows) > 0:
            #     ax.set_xlim(min(x_lows), max(x_highs))
            #     ax.set_ylim(min(y_lows), max(y_highs))
            ax.set_xlim(pc.x_low, pc.x_high)
            ax.set_ylim(pc.y_low, pc.y_high)

    p_bar.close()

    # Title
    if pc.title:
        fig.suptitle(pc.title, fontsize=30)

    # plt.tight_layout()
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    img_save_path = pc.save_path
    if img_save_path == 'auto':
        img_save_path = f'{pc.title}_{pc.tag}_{"+".join(pc.algorithms)}.png'

    img_save_path = os.path.join(IMG_SAVE_DIR, img_save_path)

    # Create the dir if not exists
    os.makedirs(os.path.dirname(img_save_path), exist_ok=True)

    plt.savefig(img_save_path)
    print(f'Saved to {img_save_path}')
    if not pc.no_show:
        plt.show()
    plt.close()
    print('Done!')


def main():
    # Get configures
    pcs = get_configs()
    for i_pc, pc in enumerate(pcs):
        print(f'Plotting an img {i_pc + 1}/{len(pcs)} (save_path={pc.save_path})')
        plot_one(pc)




if __name__ == '__main__':
    main()


