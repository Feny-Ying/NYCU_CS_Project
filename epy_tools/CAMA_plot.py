"""

@20241006 update:

- RC plot cmd:
    ```
    python epy_tools/CAMA_plot.py -d projects/CAMA/ResourceCollection/rc_sacred/ -min 950 -t test_return_mean
    ```


"""

import argparse
import os
import re
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

SKIP_CHECK_KEY = ['seed', 'set_rank_percent_instead_of_sight']


# 將 legend 提取到單獨的圖片中
def save_legend_as_image(axs, legend_file_path):
    """Create a separate legend image."""
    # Collect all handles and labels from all subplots
    handles, labels = [], []
    for ax in axs:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    # Remove duplicates by using a dictionary
    unique = {label: handle for handle, label in zip(handles, labels)}
    handles = list(unique.values())
    labels = list(unique.keys())

    # Create a separate figure for the legend
    fig, ax = plt.subplots(figsize=(6, len(labels) * 0.5))
    ax.axis('off')
    legend = ax.legend(handles, labels, loc='center', frameon=False, fontsize=12)

    # Save the legend image
    fig.savefig(legend_file_path, dpi=300, bbox_inches='tight')


# Function to parse the content and extract the training steps and values for a given tag
def extract_data_from_cout(file_content, tag):
    # Find all instances where the given tag appears with associated values
    pattern = re.compile(r't_env:\s+(\d+).*?' + re.escape(tag) + r':\s+([0-9.]+)', re.DOTALL)
    matches = pattern.findall(file_content)

    if matches:
        # Extract training steps (x) and tag values (y) from the matches
        x = np.array([int(match[0]) for match in matches])
        y = np.array([float(match[1]) for match in matches])
        return x, y
    else:
        return None, None


BASE_WIDTH = None
BASE_HEIGHT = None


def get_args():
    global BASE_WIDTH, BASE_HEIGHT
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--sacred-dir', type=str, required=True, dest='sacred_dir')
    parser.add_argument('-t', '--tag', type=str, default='test_battle_won_mean', dest='tag')
    parser.add_argument('-min', '--min-n-data', type=int, default=0, dest='min_n_data')
    parser.add_argument('-del', '--delete', default=False, dest='delete', action='store_true')
    parser.add_argument('-n', '--n-runs', type=int, default=None, dest='n_runs')
    # Optional arguments
    parser.add_argument('--line-method', type=str, default='mean', dest='line', choices=['mean', 'iqm'])
    parser.add_argument('--band-method', type=str, default='std', dest='band', choices=['std', 'ci'])
    parser.add_argument('-f', '--filter', type=str, nargs='+', default=None, dest='filter')
    parser.add_argument('-sf', '--strict-filter', type=str, nargs='+', default=None, dest='strict_filter',
                        help='Include the method if any of the strings equals to the method name')
    parser.add_argument('-se', '--strict-exclude', type=str, nargs='+', default=None, dest='strict_exclude',
                        help='Exclude the method if any of the strings equals to the method name')

    parser.add_argument('-ex', '--exclude', type=str, nargs='+', default=None, dest='exclude')
    parser.add_argument('-np', '--no-performance', default=False, dest='no_show_performance', action='store_true')
    parser.add_argument('-pre', '--prefix', type=str, default=None, dest='prefix')
    parser.add_argument('--width', type=float, default=6, dest='width')
    parser.add_argument('--height', type=float, default=6, dest='height')
    args = parser.parse_args()
    BASE_WIDTH = args.width
    BASE_HEIGHT = args.height
    return args


def process_eval_str(eval_str):
    return eval_str.replace('true', 'True').replace('false', 'False').replace('null', 'None')


TEST_UNSEEN = False


def show_data(args):
    """先列出各地圖、各方法目前的資料數量。讓人了解。"""
    global TEST_UNSEEN

    map_to_method_to_all_xy = defaultdict(lambda: defaultdict(list))

    # Check if any method string contain "test_unseen"
    test_unseen_exist = False
    for method in os.listdir(args.sacred_dir):
        if 'test_unseen' in method:
            test_unseen_exist = True
            TEST_UNSEEN = True
            break

    # First level dir: methods
    # Second level dir: maps
    # Third level dir: runs
    # Fourth level dir: data files e.g., `info.json`
    for method in os.listdir(args.sacred_dir):
        if test_unseen_exist:
            if 'test_unseen' not in method:
                raise ValueError(f'If any method contains "test_unseen", all methods should contain "test_unseen"')
        if method.startswith('_'):
            continue
        method_dir = os.path.join(args.sacred_dir, method)

        # If filter is given and none of the filter strings is in the method name, skip
        raw_method = method.replace('test_unseen', '').strip()
        if args.filter is not None and not any([f in raw_method for f in args.filter]):
            continue
        # If strict filter is given and none of the filter strings equals to the method name, skip
        if args.strict_filter is not None and not any([f == raw_method for f in args.strict_filter]):
            continue

        # If exclude is given and any of the exclude strings is in the method name, skip
        if args.exclude is not None and any([e in raw_method for e in args.exclude]):
            continue
        # If strict exclude is given and any of the exclude strings equals to the method name, skip
        if args.strict_exclude is not None and any([e == raw_method for e in args.strict_exclude]):
            continue

        for map in os.listdir(method_dir):
            map_dir = os.path.join(method_dir, map)
            valid_configs = []
            run_to_n_data = []  # List of tuple of (run_name, n_data)
            all_xy = []
            for run in os.listdir(map_dir):
                if run.startswith('_'):
                    continue
                run_dir = os.path.join(map_dir, run)
                # 若沒有 info.json，就 raise error
                if 'cout.txt' not in os.listdir(run_dir):
                    raise FileNotFoundError(f'cout.txt not found in {run_dir}')
                # Read cout.txt
                with open(os.path.join(run_dir, 'cout.txt'), 'r') as f:
                    cout = f.read()
                # Get info
                x, y = extract_data_from_cout(cout, args.tag)
                all_xy.append((x, y))

                invalid_msg = ''

                if x is not None:
                    # print(f'{run_dir: <100}: {len(x)} data points')
                    run_to_n_data.append((run, len(x)))
                    if len(x) < args.min_n_data:
                        invalid_msg = f'Has only {len(x)} data points, less than {args.min_n_data} data points'
                        raise ValueError(f'{run_dir: <100}: {invalid_msg}')
                    else:  # Valid
                        # Read `config.json`
                        with open(os.path.join(run_dir, 'config.json'), 'r') as f:
                            config = f.read()
                        valid_configs.append(config)
                else:
                    print(f'{run_dir: <100}: No data points')
                    invalid_msg = 'No data points'

                # Ask for deletion; type y or yes to delete
                if args.delete and len(invalid_msg) > 0:
                    print(f'{run_dir} is invalid: {invalid_msg}')
                    if input('Delete? (y/n) ') in ['y', 'yes']:
                        os.system(f'rm -r "{run_dir}"')
                        print(f'{run_dir} deleted')
                    else:
                        print(f'{run_dir} not deleted')

            # Check valid runs' configurations
            # 檢查是否每個run的config都一樣，若不一樣，就要標記出來
            # config 是dict。若有人的config不一樣，就要標記出來，標記方式：
            # 列不一樣的 keys
            if len(valid_configs) > 1:
                for i in range(1, len(valid_configs)):
                    if valid_configs[i] != valid_configs[0]:
                        config0 = eval(process_eval_str(valid_configs[0]))
                        config = eval(process_eval_str(valid_configs[i]))
                        for key in config0.keys():
                            # print run
                            if key not in config:
                                print(f'🔥🔥🔥 [{method}/{map} has different configurations] {key}: '
                                      f'{config0[key]} vs None')
                            if config0[key] != config[key] and key not in SKIP_CHECK_KEY:
                                print(f'🔥🔥🔥 [{method}/{map} has different configurations] {key}: '
                                      f'{config0[key]} vs {config[key]}')

            # Sort by run_names
            run_to_n_data.sort(key=lambda x: int(x[0]))
            n_data_str = ', '.join([str(n_data) for _, n_data in run_to_n_data])
            run_name_str = ', '.join([run_name for run_name, _ in run_to_n_data])
            print(f'{map_dir: <120}: {len(run_to_n_data)} runs -> n_data: {n_data_str:50} (run names: {run_name_str})')

            # Cut by the min n_data
            real_min_n_data = min([n_data for _, n_data in run_to_n_data])
            try:
                cut_all_xy = [(x[:real_min_n_data], y[:real_min_n_data]) for x, y in all_xy]
            except TypeError as e:
                print(f'Processing: {map_dir}')
                raise e

            #
            map_to_method_to_all_xy[map][method] = cut_all_xy
    return map_to_method_to_all_xy


def main():
    args = get_args()
    map_to_method_to_all_xy = show_data(args)

    # Sort maps
    # Each map has a score, larger score means the map is going to be shown later
    # If the name is not in the map_score, it will be assigned a score of 4
    map_score = {
        '3-8sz_symmetric': 1,
        '3-8csz_symmetric': 2,
        '3-8MMM_symmetric': 3, }
    map_to_method_to_all_xy = dict(sorted(map_to_method_to_all_xy.items(), key=lambda x: map_score.get(x[0], 4)))

    XY_LABEL_FONT_SIZE = 20
    TITLE_FONT_SIZE = 24
    LEGEND_FONT_SIZE = 15
    TICK_SIZE = 12

    plt.style.use('ggplot')
    # plt.style.use('seaborn-paper')
    # plt.style.use('seaborn-v0_8-paper')
    # plt.style.use('seaborn-v0_8-whitegrid')
    # plt.style.use('seaborn-v0_8-darkgrid')
    # plt.style.use('classic')

    GLOBAL_FONT_SERIF = ['DejaVu Serif']
    GLOBAL_FONT_BASE = {'family': 'serif'}
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = GLOBAL_FONT_SERIF
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["mathtext.rm"] = "serif"  # 確保數字部分也是使用襯線字體

    # Use subplot; a row of subplots (each subplot is a map)
    n_maps = len(map_to_method_to_all_xy)
    base_height = BASE_HEIGHT
    base_width = BASE_WIDTH
    fig, axs = plt.subplots(1, n_maps, figsize=(base_height * n_maps, base_width))
    if n_maps == 1:
        axs = [axs]
    for i, (map, method_to_all_xy) in enumerate(map_to_method_to_all_xy.items()):
        ax = axs[i]
        for method, all_xy in method_to_all_xy.items():
            # Aggregate all runs' xy data
            # Let's use mean and std band
            xs = [x for x, _ in all_xy]
            ys = [y for _, y in all_xy]
            # Calculate mean and std
            if args.line == 'mean':
                line_y = np.mean(ys, axis=0)
                if not args.no_show_performance:
                    print(f'{method} {map} last mean: {line_y[-1]}', end='\t')
            else:
                raise NotImplementedError(f'line method {args.line} not implemented')
            if args.band == 'std':
                band_y = np.std(ys, axis=0)
                if not args.no_show_performance:
                    print(f'std: {band_y[-1]}')
            else:
                raise NotImplementedError(f'band method {args.band} not implemented')
            # Plot
            label = f'{method} (l={len(ys[0])}, n={len(ys)})'

            # --- Style ---
            linestyle = '-'
            color = None

            raw_method = method.replace('test_unseen', '').strip()

            # # Line style
            # if any([s in method for s in ['3s', '6s']]):
            #     linestyle = '--'
            # elif any([s in method for s in ['0.4', '0.8']]):
            #     linestyle = '-.'

            NAME_VERSIONS = [
                'add',  # 用加的方式呈現。例如 IM-Qatten + 某個CAMA部件 + DSR
                'paper',  # 要秀給人看，沒要寫太細
                'both',  # 兩個都要
            ]
            # Pick up one
            # NAME_VERSION = 'add'
            NAME_VERSION = 'paper'
            # NAME_VERSION = 'both'

            assert NAME_VERSION in NAME_VERSIONS
            z_order = 2.5
            legend_order = 0
            if raw_method in ['qmix_atten']:
                method_add = 'Qatten'
                # method_paper = 'Qatten $d=9$'
                method_paper = 'Qatten'
                color = 'gray'
            elif raw_method in ['icm_qmix_atten_nomi']:
                method_add = 'IM-Qatten'
                # method_paper = 'IM-Qatten $d=9$'
                method_paper = 'IM-Qatten'
                color = 'black'
            elif raw_method in ['icm_qmix_atten']:
                method_add = 'IM-Qatten + GlobalCoach'
                # method_paper = 'IM-Qatten $d=9$ + CAMA w/o AWR'
                method_paper = 'IM-Qatten + CAMA w/o AWR'
            elif raw_method in ['dsr_icm_qmix_atten_nomi']:
                method_add = 'IM-Qatten + DSR'
                method_paper = 'IM-Qatten + DSR (ours)'
                color = 'red'
                z_order = 3.2
            elif raw_method in ['dsr_icm_qmix_atten_nomi_6s']:
                method_add = 'IM-Qatten + 6s'
                method_paper = 'IM-Qatten $d=6$'
                color = 'yellow'
            elif raw_method in ['dsr_icm_qmix_atten_nomi_3s']:
                method_add = 'IM-Qatten + 3s'
                method_paper = 'IM-Qatten $d=3$'
                color = 'orange'
            elif raw_method in ['icm_qmix_atten_a8']:
                method_add = 'IM-Qatten + AWR + GlobalCoach'
                # method_paper = 'IM-Qatten $d=9$ + CAMA'
                method_paper = 'IM-Qatten + CAMA'
                color = 'C1'

                z_order = 3

            elif raw_method in ['icm_qmix_atten_nomi_a8']:
                method_add = 'IM-Qatten + AWR'
                # method_paper = 'IM-Qatten $d=9$ + CAMA w/o GlobalCoach'
                method_paper = 'IM-Qatten + CAMA w/o GlobalCoach'
                color = 'C4'
            elif raw_method in ['dsr_icm_qmix_atten_a8']:
                method_add = 'IM-Qatten + AWR + GlobalCoach + DSR'
                method_paper = 'IM-Qatten + CAMA + DSR (ours)'
                color = '#6D8C3E'
                z_order = 3.4

            elif raw_method in ['dsr_icm_qmix_atten_nomi_a8']:
                method_add = 'IM-Qatten + AWR + DSR'
                method_paper = 'IM-Qatten + CAMA w/o GlobalCoach + DSR (ours)'
            elif raw_method in ['dsr_icm_qmix_atten']:
                method_add = 'IM-Qatten + GlobalCoach + DSR'
                method_paper = 'IM-Qatten + CAMA w/o AWR + DSR (ours)'
                color = 'C2'

            else:
                raise NotImplementedError(f'raw_method {raw_method} not implemented')

            print(f'{str(method):<50} {"(" + str(raw_method) + ")":<50}: {str(color):>30}')

            if NAME_VERSION == 'add':
                label_name = method_add
            elif NAME_VERSION == 'paper':
                label_name = method_paper
            elif NAME_VERSION == 'both':
                if method_add == method_paper:
                    label_name = f'{method_add}'
                else:
                    # label_name = f'{method_add} ({method_paper})'
                    label_name = f'{method_paper} ({method_add})'
            else:
                raise NotImplementedError(f'NAME_VERSION {NAME_VERSION} not implemented')

            the_line, = ax.plot(xs[0], line_y, label=label_name, color=color, linestyle=linestyle, zorder=z_order)
            ax.fill_between(xs[0], line_y - band_y, line_y + band_y, alpha=0.1, color=the_line.get_color(),
                            zorder=z_order)

            # x y labels
            ax.set_xlabel('Steps')
            y_label = None
            if args.tag == 'test_return_mean':
                y_label = 'Test Return'
            elif args.tag == 'test_battle_won_mean':
                y_label = 'Test Win Rate'
            elif args.tag == 'test_visibility_mean':
                # ax.set_ylabel('Visibility Ratio')
                pass
            elif args.tag == 'test_selected_sight_mean':
                y_label = 'Test Selected Sight'
            ax.set_ylabel(y_label, fontsize=XY_LABEL_FONT_SIZE)

            """
            
            python epy_tools/CAMA_plot.py -d projects/CAMA/sacred_FORMAL_SMAC-DT/fff -min 970 -f nomi -ex "ucb" -t test_battle_won_mean -del ; \
            python epy_tools/CAMA_plot.py -d projects/CAMA/sacred_FORMAL_SMAC-DT/fff -min 970 -f nomi -ex "ucb" -t test_return_mean -del ; \
            python epy_tools/CAMA_plot.py -d projects/CAMA/sacred_FORMAL_SMAC-DT/fff -min 970 -f "nomi dsr(3s" -t test_selected_sight_mean -del  
            
            python epy_tools/CAMA_plot.py -d projects/CAMA/sacred_FORMAL_SMAC-DT/fff -min 970 -f nomi -ex "ucb" -t test_battle_won_mean &&
            python epy_tools/CAMA_plot.py -d projects/CAMA/sacred_FORMAL_SMAC-DT/fff -min 970 -f nomi -ex " 6s" " 3s" "ucb" -t test_battle_won_mean
            
            # 2 lines
            
            python epy_tools/CAMA_plot.py -d projects/CAMA/sacred_FORMAL_SMAC-DT/fff -min 970 -f nomi -ex "ucb" -t test_battle_won_mean && \
            python epy_tools/CAMA_plot.py -d projects/CAMA/sacred_FORMAL_SMAC-DT/fff -min 970 -f nomi -ex " 6s" " 3s" "ucb" ".4" "ori" -t test_battle_won_mean; \
            python epy_tools/CAMA_plot.py -d projects/CAMA/sacred_FORMAL_SMAC-DT/fff -min 970 -f nomi -ex " 6s" " 3s" "ucb" ".4" "ori" -t test_return_mean; \
            python epy_tools/CAMA_plot.py -d projects/CAMA/sacred_FORMAL_SMAC-DT/fff -min 970 -f nomi -ex " 6s" " 3s" "ucb" ".4" "ori" ".8" "nomi dsr(3s" -t test_selected_sight_mean

            python epy_tools/CAMA_plot.py -d projects/CAMA/sacred_FORMAL_SMAC-DT/fff -min 970  -t test_battle_won_mean --no-performance -del -ex  "3s" "a4" "6s" -del
            
            # ======== 畫 UnseenTrue =========
            
            # Basic: 比較 baseline vs +DSR vs +CAMA vs +CAMA+DSR
            python epy_tools/CAMA_plot.py -d projects/CAMA/sacred_testUnssen/ -t test_battle_won_mean --no-performance -del   -del -min 970  -sf "icm_qmix_atten_nomi" "dsr_icm_qmix_atten_nomi" "icm_qmix_atten_a8" "dsr_icm_qmix_atten_a8"  -pre basic

            # 同上，但印 selected sight
            python epy_tools/CAMA_plot.py -d projects/CAMA/sacred_testUnssen/ -t test_selected_sight_mean --no-performance -del   -del -min 970  -sf "dsr_icm_qmix_atten_nomi" "dsr_icm_qmix_atten_a8"  -pre basic
            
            # 更多 CAMA 的 ablation 或 +DSR
            python epy_tools/CAMA_plot.py -d projects/CAMA/sacred_testUnssen/ -t test_battle_won_mean --no-performance -del   -del -min 970  -sf "icm_qmix_atten_nomi" "dsr_icm_qmix_atten_nomi" "icm_qmix_atten_a8" "dsr_icm_qmix_atten_a8" "icm_qmix_atten_nomi_a8" "dsr_icm_qmix_atten" -pre more_cama --width 8 --height 8

            # 同上，但印 selected sight
            python epy_tools/CAMA_plot.py -d projects/CAMA/sacred_testUnssen/ -t test_selected_sight_mean --no-performance -del  -min 970  -sf  "dsr_icm_qmix_atten_nomi"  "dsr_icm_qmix_atten_a8"  "dsr_icm_qmix_atten" -pre more_cama  --width 8 --height 8

            # Pure Sights
            python epy_tools/CAMA_plot.py -d projects/CAMA/sacred_testUnssen/ -t test_battle_won_mean --no-performance -del   -del -min 970  -sf "icm_qmix_atten_nomi" "dsr_icm_qmix_atten_nomi" "icm_qmix_atten_a8" "dsr_icm_qmix_atten_nomi_3s" "dsr_icm_qmix_atten_nomi_6s" -pre  pure            
            
            # Qatten vs IM-Qatten
            python epy_tools/CAMA_plot.py -d projects/CAMA/sacred_testUnssen/ -t test_battle_won_mean --no-performance -del   -del -min 970  -sf "icm_qmix_atten_nomi" "dsr_icm_qmix_atten_nomi" "icm_qmix_atten_a8" "qmix_atten"  -pre qatten
                        
             
            # # ======== 畫 UnseenFalse =========
            # 
            # Basic: 比較 baseline vs +DSR vs +CAMA vs +CAMA+DSR
            python epy_tools/CAMA_plot.py -d projects/CAMA/sacred_FORMAL_SMAC-DT/fff -t test_battle_won_mean --no-performance -del   -del -min 970  -sf "icm_qmix_atten_nomi" "dsr_icm_qmix_atten_nomi" "icm_qmix_atten_a8" "dsr_icm_qmix_atten_a8"  -pre basic

            # 更多 CAMA 的 ablation 或 +DSR
            python epy_tools/CAMA_plot.py -d projects/CAMA/sacred_FORMAL_SMAC-DT/fff -t test_battle_won_mean --no-performance -del   -del -min 970  -sf "icm_qmix_atten_nomi" "dsr_icm_qmix_atten_nomi" "icm_qmix_atten_a8" "dsr_icm_qmix_atten_a8" "icm_qmix_atten_nomi_a8" "dsr_icm_qmix_atten" -pre more_cama

            # 同上，但印 selected sight
            python epy_tools/CAMA_plot.py -d projects/CAMA/sacred_FORMAL_SMAC-DT/fff -t test_selected_sight_mean --no-performance -del  -min 970  -sf  "dsr_icm_qmix_atten_nomi"  "dsr_icm_qmix_atten_a8"  "dsr_icm_qmix_atten" -pre more_cama

            # Pure Sights
            python epy_tools/CAMA_plot.py -d projects/CAMA/sacred_FORMAL_SMAC-DT/fff -t test_battle_won_mean --no-performance -del   -del -min 970  -sf "icm_qmix_atten_nomi" "dsr_icm_qmix_atten_nomi" "icm_qmix_atten_a8" "dsr_icm_qmix_atten_nomi_3s" "dsr_icm_qmix_atten_nomi_6s" -pre  pure            

            # Qatten vs IM-Qatten
            python epy_tools/CAMA_plot.py -d projects/CAMA/sacred_FORMAL_SMAC-DT/fff -t test_battle_won_mean --no-performance -del   -del -min 970  -sf "icm_qmix_atten_nomi" "dsr_icm_qmix_atten_nomi" "icm_qmix_atten_a8" "qmix_atten"  -pre qatten
               
            
            """

        CHANGE_3_8_to_3_5 = True
        CHANGE_3_8_to_3_5 = False
        if CHANGE_3_8_to_3_5:
            map = map.replace('3-8', '3-5')

        NO_SHOW_NUMBER = False
        if NO_SHOW_NUMBER:
            map = map.replace('3-5', '').replace('3-8', '')

        ax.set_title(map, fontsize=TITLE_FONT_SIZE)
        # ax.legend()

        # Tick size
        ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
        ax.xaxis.get_offset_text().set_fontsize(TICK_SIZE)  # 調整 x 軸的 offset label 字體大小
        ax.yaxis.get_offset_text().set_fontsize(TICK_SIZE)  # 調整 y 軸的 offset label 字體大小

        # Sort legend
        handles, labels = ax.get_legend_handles_labels()

        # 定義自訂排序邏輯：用 legend_order 來排序
        LEGEND_ORDERS = {
            'Qatten': 0,
            'IM-Qatten': 1,
            'IM-Qatten + DSR (ours)': 2,
            'IM-Qatten + DSR': 2,
        }

        # 定義自訂排序邏輯：用 legend_order 來排序
        labels_handles = zip(labels, handles)
        labels_handles = sorted(
            labels_handles,
            key=lambda x: (LEGEND_ORDERS.get(x[0], 100), x[0])
        )

        # key = lambda x: (0 if x[0].startswith('Qatten') else 1, x[0])  # Qatten 開頭優先，其他按字典序

        # 解壓排序後的結果
        labels, handles = zip(*labels_handles)

        # 設置圖例
        ax.legend(handles, labels, fontsize=LEGEND_FONT_SIZE)

    # title = f'{args.tag} (line: {args.line}, band: {args.band}))'
    # fig.suptitle(title)

    # Save
    time_str = f'{datetime.now():%Y%m%d_%H%M%S}'
    save_plt_dir = 'exp_plot/CAMA'
    os.makedirs(save_plt_dir, exist_ok=True)

    # Tight
    plt.tight_layout()

    # plt.savefig(os.path.join(save_plt_dir, f'{time_str}.png'))
    # plt.savefig(os.path.join(save_plt_dir, f'{args.tag}.png'), dpi=300)
    if args.prefix is None:
        filename = f'{args.tag}.pdf'
    else:
        filename = f'{args.prefix} {args.tag}.pdf'
    if TEST_UNSEEN:
        filename = f'TestUnseen {filename}'

    plt.savefig(os.path.join(save_plt_dir, filename), dpi=300)
    # # 保存 legend 為單獨的圖片
    #  if args.prefix in ['basic']:
    #      legend_file_path = os.path.join(save_plt_dir, 'legend.pdf')
    #      save_legend_as_image(axs, legend_file_path)


if __name__ == '__main__':
    main()
