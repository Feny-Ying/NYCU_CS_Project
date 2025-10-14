# noinspection PyPackageRequirements
import argparse
import json
import os
import re
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import product
from pprint import pprint
from typing import Optional, Union, Any

import numpy as np
import pandas as pd
import yaml
from matplotlib import font_manager
from matplotlib import pyplot as plt
# from scipy.stats import stats
from scipy import stats
from tqdm import tqdm

# Basic settings
MUST_HAVE_KEYS = ['parent']  # Each config must have
OPTIONAL_KEYS = ['domain', 'hp_all_skip',
                 # 以下用來規範允許的超參數值。請用 dict, value 為 list of allowed values
                 'hp_all',
                 'hp_alg_group',
                 'hp_alg',
                 'hp_map',
                 'hp_sight',
                 'hp_map_sight',

                 'hp_other_variables', 'seeds', 'plot', 'metrics_json_name']
VAR_DELIMITER = ' '

# Constants for plotting
BAND_NAMES = {'ci': '95%CI', 'stderr': 'stderr', 'minmax': 'min-max', 'std': 'std'}
DEFAULT_IMG_SAVE_DIR = 'exp_plot'

#
METRICS_JSON_NAME = 'metrics.json'


# noinspection PyUnresolvedReferences
@dataclass(frozen=True)
class ExpVarComb:
    """一個便於後面使用的資料結構，用來表示一個實驗的變數組合。"""
    params: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def __post_init__(self):
        """將 params 裡的變數都變成物件的屬性。讓它可以用 obj.key 的方式取值。"""
        object.__setattr__(self, "_params", dict(self.params))
        for key, value in self._params.items():
            object.__setattr__(self, key, value)

    def get_method_name(self, all_exp_vars: list[str], skip_map=True, skip_vars: Optional[list[str]] = None):
        """這裡的 method_name 指的是由 alg + sight + ... 所組成在 sacred 下的 exp name。
        注意：map (嚴格講是其對應的 env_id) 是在 sacred/method_name/ 下。"""
        if skip_vars is None:
            skip_vars = []
        if skip_map:
            skip_vars.append('map')
        skip_vars = set(skip_vars)
        return VAR_DELIMITER.join([self[k] for k in all_exp_vars if k not in skip_vars])

    def __getitem__(self, item):
        """讓其和 dict 一樣可以用 [] 取值。"""
        return self._params[item]
        # return

    def keys(self):
        """讓其和 dict 一樣可以用 keys() 取值。"""
        return self._params.keys()

    def items(self):
        """讓其和 dict 一樣可以用 items() 取值。"""
        return self._params.items()

    def values(self):
        """讓其和 dict 一樣可以用 values() 取值。"""
        return self._params.values()

    def __repr__(self):
        # return f'ExpVarComb with keys: {self._params.keys()}'
        return self.__str__()

    def __str__(self):
        """Format: ExpVarComb(alg=qmix, map=8m_vs_9m, sight=6, rs=rs1)"""
        return f'ExpVarComb({", ".join([f"{k}={v}" for k, v in self.params])})'

    @staticmethod
    def get_exp_var_comb(var_dict: dict):
        """給定一個 exp_vars 的 dict，將它轉換成一個 ExpVarComb 物件。
        Input 範例：dict(alg='qmix', map='8m_vs_9m', sight=6, rs='rs1')
        當然詳細格式是要看``ExpVarComb``的定義的。
        """
        # Make all values to str
        for k, v in var_dict.items():
            var_dict[k] = str(v)
        var_tuple = tuple(var_dict.items())
        # Sort by key (tuple[0] is key)
        var_tuple = tuple(sorted(var_tuple, key=lambda x: x[0]))
        comb = ExpVarComb(var_tuple)
        return comb


@dataclass
class PlotConfig:  # "pc": abbreviation of `PlotConfig`
    # The default is an example for plotting test_return_mean

    # Save path (if None or empty, do not save img) (if 'auto', it will automatically produce a name)
    # 新功能：若 sav_path 的結尾是 "/"，那麼檔案自動用 title 來命名
    save_path: Optional[str] = None
    # ---------- Data ----------
    # Path to the sacred dir. Use argparse to get this if not given in the yaml
    # Recommendation: not given in the yaml
    f: Optional[str] = None
    # Title
    title: Optional[str] = None
    # Tag to plot
    tag: str = 'test_return_mean'
    # Number of data to plot; if None, get min length of all runs
    # If 'auto':
    #            if tag == 'test_return_mean', use the `n_min_tests` specified in the config yaml
    #            if other tags, use the min length of all runs
    n_tag_data: Optional[Union[int, str]] = 'auto'
    # Available: 'ci' (95%ci), 'std', 'stderr', 'minmax' (point is mean) # TODO
    # band: str = 'ci'
    band: str = 'std'
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
    y_scale: str = 'linear'
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
    label_without_len_and_runs: bool = False
    hide_subtitle_description: bool = False
    show_title: bool = True
    show_subtitle: bool = True
    show_legend: bool = True
    show_text_on_line: bool = True
    # Label 上只寫除了 col row 以外的變數
    custom_ax_row_var: str = 'alg'
    custom_ax_col_var: str = 'map'
    row_var_order: Optional[list[str]] = None
    col_var_order: Optional[list[str]] = None
    row_upper: bool = False
    col_upper: bool = False
    legend_loc: Optional[str] = None
    # 若有給，則符合的 alg (dict 第一層 key 為關鍵字)其的 x/y low/high (第二層 key 可以放的： y_low, y_high, x_low, x_high)
    # 調整到這個值
    map_keyword_to_low_or_high: Optional[dict[str, dict[str, str]]] = None
    sight_keyword_to_low_or_high: Optional[dict[str, dict[str, str]]] = None
    # 關閉顯示 row 名或 col 名
    no_show_row_names: bool = False
    no_show_col_names: bool = False
    # 讓長的像"{digit}s" 名字都改成 w/o DSR
    change_pure_sight_name_to_no_dsr: bool = False
    change_ucb_name_to_dsr: bool = False
    change_ucb_name_to_ucb_with_sights: bool = False
    change_ucb_name_to_original: bool = False
    change_ucb_name_by_mask_assuming_only_mask_both: bool = False
    # even if using "change_ucb_name_by_mask_assuming_only_mask_both", not adding w/ or w/o...
    change_ucb_name_by_mask_assuming_only_mask_both_not_showing_with_or_without: bool = False
    # Row name padding
    row_title_pad: int = 65
    # Sort legend by suffix digits
    sort_legend_by_suffix_digits: bool = False
    # Sort legend by alphabet
    sort_legend_by_alphabet: bool = False
    # If True, output each run's data to a file
    output_each_run_data: bool = False
    # Use IQM
    use_iqm: bool = False
    # Force col names
    force_col_names: Optional[list[str]] = None
    # Please no d for non-ucb
    no_show_d_for_non_ucb: bool = False


def get_mean_func(pc: PlotConfig, axis: Optional[int] = 0):
    if pc.use_iqm:  # Use Interquartile Mean (IQM)
        # Remove outliers (25% and 75% quantile) and compute mean
        def mean_func(data):
            data = np.array(data)
            n_to_remove = int(data.shape[0] * 0.25)
            iqm = np.mean(np.sort(data, axis=0)[n_to_remove:-n_to_remove], axis=axis)
            return iqm
    else:
        def mean_func(data):
            return np.mean(data, axis=axis)

    return mean_func


def load_yaml(yaml_path: str) -> dict:
    """Read data from a yaml file."""
    with open(yaml_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


# 定義一個函數來從每個 label 中提取數字，並作為排序的鍵
def extract_number_from_label(label):
    # 如果 label 以 $ 結尾，移除最後一個字符
    if label.endswith('$'):
        label = label[:-1]

    # 從末尾向前尋找一個數字 (包括可能的小數點)
    match = re.search(r'(\d+(\.\d+)?)', label)
    if match:
        result = float(match.group(1))  # 如果找到數字，轉換為 float
    else:
        result = float('inf')  # 如果沒有找到數字，返回正無限大
    return result


def get_args():
    parser = argparse.ArgumentParser()
    # Required
    parser.add_argument('-f', '--folder', type=str, help='Path to the Sacred folder', required=True)
    parser.add_argument('-c', '--config', type=str, help='Path to the config yaml file', required=True)
    parser.add_argument('-save', '--save', type=str, help='Folder to save the plots', default=DEFAULT_IMG_SAVE_DIR)
    # Optional
    parser.add_argument('-no-check', '--no-check', action='store_true', help='Skip the check of the runs')
    parser.add_argument('-np', '--no_plot', action='store_true', help='Do not plot the results, only check them')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print more information')
    # Useful tools
    parser.add_argument('-del', '--delete', action='store_true', help='Delete the invalid runs')
    return parser.parse_args()


def get_env_id(config: dict, hp_constraints: dict) -> str:
    """給定一個 comb，回傳它的 env_id。"""
    # Get env_id (sc2 和 gymma 不一樣)
    if len(config['hp_all']['env']) != 1:
        raise ValueError(f'``env`` should only have one value.')
    env_name = config['hp_all']['env'][0]
    if env_name in ['sc2', 'asc2']:
        env_id_list = hp_constraints['env_args']['map_name']
    elif env_name in ['gymma']:
        env_id_list = hp_constraints['env_args']['key']
    elif env_name in ['sc2custom']:
        env_id_list = hp_constraints['scenario']
    else:
        raise ValueError(f'Unknown env: {env_name}')
    if len(env_id_list) != 1:
        print(hp_constraints['env_args'])
        raise ValueError(f'env_id_list should contain only one element, but got {env_id_list}')
    env_id = env_id_list[0]
    return env_id


def smac_name_replacement(string):
    if string == 'originalState':
        string = 'w/ Given State'
    elif string == 'concatObsAsState':
        string = 'w/o Given State'
    return string


def map_name_unifying(string, var_name, config):
    """為了要統一把map name的 "?x" 放在名稱最後。"""
    if var_name != 'map':
        return string
    domain = config['domain']
    map_id = string

    if domain == 'LBF':
        if 's-' in map_id and map_id.split('s-')[0].isdigit():
            s = map_id.split('s-')[0]
            rest = map_id.split('s-')[1]
            return f'{rest}-{s}s'
    elif domain == 'RWARE':
        return string
    elif domain == 'SMAC':
        pass

    # print(f'domain: {config["domain"]}')
    # print(f'string: {string}')
    # raise NotImplementedError

    return string


# noinspection DuplicatedCode
def plot_one(pc: PlotConfig, config: dict, combinations: set[ExpVarComb], all_exp_vars: list[str],
             comb_to_hp_constraints: dict[ExpVarComb, dict[str, str]], args: argparse.Namespace):
    """"畫一張圖

    Subplots 的方式來畫。多個小圖組成一張大圖。
    大圖的 row 是 alg，col 是 map。
    """
    plt.rcParams['figure.dpi'] = pc.dpi
    plt.style.use('ggplot')

    # plt.rcParams['grid.color'] = 'gray'  # 修改網格線顏色
    # plt.rcParams['grid.alpha'] = 0.1  # 修改網格線透明度，值越小越淡（0 到 1 之間）

    # 檢查 ax 的 col & row 的 vars 設定
    row_var = pc.custom_ax_row_var
    col_var = pc.custom_ax_col_var

    # Get all the env & method dir (since assuming the `check_exp.py` is done, we can directly derive the paths
    # using the `f` and `config` yaml)

    # Get all possible algs and maps
    row_var_values = set()
    col_var_values = set()
    all_var_values = defaultdict(set)
    for combination in combinations:
        row_var_values.add(combination[row_var])
        col_var_values.add(combination[col_var])
        for var in all_exp_vars:
            all_var_values[var].add(combination[var])
    n_rows, n_cols = len(row_var_values), len(col_var_values)
    row_var_values, col_var_values = list(row_var_values), list(col_var_values)

    # 算各個 var 的 unique value 數量
    all_var_n_unique_values = {var: len(all_var_values[var]) for var in all_var_values}
    only_one_unique_value_var_and_values = {var: list(all_var_values[var])[0] for var, n_unique_values in
                                            all_var_n_unique_values.items() if
                                            n_unique_values == 1}

    # 將 row var & col var 排成我們要的順序
    if pc.row_var_order is not None:
        row_var_values = sorted(row_var_values,
                                key=lambda x: pc.row_var_order.index(x) if x in pc.row_var_order else 999)
    if pc.col_var_order is not None:
        col_var_values = sorted(col_var_values,
                                key=lambda x: pc.col_var_order.index(x) if x in pc.col_var_order else 999)

    # =============== Subplot 設定 =================
    print(f'Plotting {n_rows} rows ({row_var}) and {n_cols} cols ({col_var})')
    try:
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * pc.base_width, n_rows * pc.base_height))
    except ValueError as e:
        print(f'Error when creating subplots: {e}')
        print(f'row_var_values: {row_var_values}')
        print(f'col_var_values: {col_var_values}')
        print(f'combinations: {combinations}')
        print(f'all_exp_vars: {all_exp_vars}')
        raise e

    # =========== 根據 domain 改全局的風格 ===========

    GLOBAL_FONT_SERIF = ['DejaVu Serif']
    GLOBAL_FONT_BASE = {'family': 'serif'}
    LEGEND_ZORDER = 11

    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["mathtext.rm"] = "serif"  # 確保數字部分也是使用襯線字體

    GLOBAL_FONT_TITLE = deepcopy(GLOBAL_FONT_BASE)
    GLOBAL_FONT_XY_LABEL = deepcopy(GLOBAL_FONT_BASE)

    # noinspection PyTypeChecker
    GLOBAL_FONT_TITLE.update({'size': 24})
    # noinspection PyTypeChecker
    GLOBAL_FONT_XY_LABEL.update({'size': 20})
    COL_NAME_FONT_SIZE = 24
    ROW_NAME_FONT_SIZE = 24
    LEGEND_FONT_SIZE = 15

    # 全局設置 x 和 y 軸刻度標籤的字體大小
    TICK_SIZE = 12

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = GLOBAL_FONT_SERIF

    if config['domain'] == 'LBF':
        pass

    # ==================================

    # 設好大圖的 row 和 col 的標題
    # Reference: https://stackoverflow.com/questions/25812255/row-and-column-headers-in-matplotlibs-subplots
    row_title_pad = pc.row_title_pad  # in points
    col_title_pad = 15  # in points
    col_titles = list(map(lambda x: x.strip().upper() if pc.col_upper else x.strip(), col_var_values))
    if pc.force_col_names is not None:
        col_titles = pc.force_col_names
    row_titles = list(map(lambda x: x.strip().upper() if pc.row_upper else x.strip(), row_var_values))
    if len(col_titles) > 1 and len(row_titles) > 1:
        axs_for_titles = axs
    elif len(col_titles) > 1 and len(row_titles) == 1:
        axs_for_titles = axs.reshape(1, -1)
    elif len(col_titles) == 1 and len(row_titles) > 1:
        axs_for_titles = axs.reshape(-1, 1)
    else:  # Only one row and one col
        axs_for_titles = np.array([[axs]])
    if not pc.no_show_col_names:
        for ax, col in zip(axs_for_titles[0, :], col_titles):
            if config['domain'] == 'SMAC':
                col = smac_name_replacement(col)
            col = map_name_unifying(col, col_var, config)
            ax.annotate(col, xy=(0.5, 1), xytext=(0, col_title_pad),
                        xycoords='axes fraction', textcoords='offset points',
                        size=COL_NAME_FONT_SIZE, ha='center', va='baseline')
            # size='xx-large', ha='center', va='baseline')
    if not pc.no_show_row_names:
        for ax, row in zip(axs_for_titles[:, 0], row_titles):
            if config['domain'] == 'SMAC':
                row = smac_name_replacement(row)
            row = map_name_unifying(row, row_var, config)
            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - row_title_pad, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        size=ROW_NAME_FONT_SIZE, ha='left', va='center')
            # size='xx-large', ha='left', va='center')

    # =============== 各小圖繪製 =================
    # Starting the loop to plot each grid in the subplot
    p_bar = tqdm(total=len(combinations), desc='Plotting each data')

    # Init a dict for keep all last mean & std (for each subplot, method_name -> mean+-std)
    row_var_to_col_var_to_value_dict = defaultdict(lambda: defaultdict(dict))

    for i_comb, comb in enumerate(combinations):
        row_var_value = comb[row_var]
        col_var_value = comb[col_var]
        method_name = comb.get_method_name(all_exp_vars)

        # 對於沒有當 row 或 col 的變數，若它的數量只有一個，可以不用在 label 上顯示，但加在 title_description
        skip_vars = [row_var, col_var] + list(only_one_unique_value_var_and_values.keys())
        var_str_without_row_col = comb.get_method_name(all_exp_vars, skip_vars=skip_vars)
        var_str_without_row_col = var_str_without_row_col.strip()
        if var_str_without_row_col == '':  # 為避免啥都沒有
            var_str_without_row_col = comb.get_method_name(all_exp_vars, skip_vars=None)

        # ==============  要畫圖在 paper 上的話，名字可能要改一下。 ===================

        try:
            is_pure_sight = 'ucb' not in var_str_without_row_col and (
                    comb['sight'].replace('masia_', '')[-1] == 's' and comb['sight'].replace(
                'masia_', '')[:-1].isdigit())
        except IndexError as e:
            print(f'var_str_without_row_col: {var_str_without_row_col}, comb: {comb}')
            print(var_str_without_row_col[-1])
            print(var_str_without_row_col[:-1])
            raise e

        use_masia = False
        if 'masia' in var_str_without_row_col:
            use_masia = True
            var_str_without_row_col = var_str_without_row_col.replace('masia_', '').replace(
                'masia', '').strip()
            print(f'var_str_without_row_col: {var_str_without_row_col}, comb: {comb}')


        if is_pure_sight:

            if pc.change_pure_sight_name_to_no_dsr:
                if 'pure' in var_str_without_row_col:
                    if pc.change_ucb_name_to_dsr:
                        digit = comb['sight'].replace('masia_','')[:-1]
                        assert digit.isdigit()
                        var_str_without_row_col = f'{comb["alg"].upper()} $d$={digit:>2}'
                    else:
                        var_str_without_row_col = f'{comb["alg"].upper()}'
                elif pc.change_ucb_name_to_original:
                    FORCE_HAVE_SR = True
                    if pc.no_show_d_for_non_ucb:
                        FORCE_HAVE_SR = False
                    if FORCE_HAVE_SR:
                        digit = comb['sight'].replace('masia_','')[:-1]
                        assert digit.isdigit()
                        var_str_without_row_col = f'{comb["alg"].upper()} $d$={digit:>2}'
                    else:
                        var_str_without_row_col = f'{comb["alg"].upper()}'
                elif 'MLP' in var_str_without_row_col or 'GRU' in var_str_without_row_col:
                    is_mlp = 'MLP' in var_str_without_row_col
                    var_str_without_row_col = var_str_without_row_col.replace('MLP', '').replace('GRU', '').strip()
                    digit = comb['sight'].replace('masia_','')[:-1]
                    assert digit.isdigit()
                    var_str_without_row_col = f'{"MLP" if is_mlp else "GRU"} $d$={digit:>2}'
                elif pc.change_ucb_name_to_dsr:
                    if pc.no_show_d_for_non_ucb:
                        var_str_without_row_col = f'{comb["alg"].upper()}'
                    else:
                        digit = var_str_without_row_col[:-1]
                        var_str_without_row_col = f'{comb["alg"].upper()} $d$={digit:>2}'
                else:
                    var_str_without_row_col = f'{comb["alg"].upper()}'
            # else:
            #     digit = var_str_without_row_col[:-1]
            #     var_str_without_row_col = f'SR={digit:>2}'
        elif 'DSR_' in var_str_without_row_col and isinstance(var_str_without_row_col, str):
            # If start with DSR_c and the rest is float or int
            if var_str_without_row_col.startswith('DSR_c') and var_str_without_row_col[5:].replace(
                    '.', '').isdigit():
                # 對齊在小數點，並且字串同長
                c_value = f'{float(var_str_without_row_col[5:]):>5}'
                if c_value in ['99999999999', '99999999999.0']:
                    c_value = r'$\ \ \ \ \infty$'
                # Make latex $c=xxx$
                var_str_without_row_col = f'$c$={c_value}'
            elif var_str_without_row_col.startswith('DSR_w') and var_str_without_row_col[5:].replace(
                    '.', '').isdigit():
                w_value = f'{int(var_str_without_row_col[5:]):>6}'
                if w_value in ['99999999999', '99999999999.0', ]:
                    w_value = r'$\ \ \ \ \ \infty$'
                var_str_without_row_col = f'$w$={w_value}'
            elif var_str_without_row_col.startswith('DSR_ucb'):
                var_str_without_row_col = var_str_without_row_col.replace(
                    # 'DSR_ucb', '$\mathcal{R}$=').replace(
                    'DSR_ucb', '$\mathcal{D}$=').replace(
                    '(', '{').replace(')', '}')
            elif var_str_without_row_col.startswith('DSR_') and 'Sight' in var_str_without_row_col:
                var_str_without_row_col = var_str_without_row_col.replace(
                    'DSR_addSightOneHot', 'w/  sight ID').replace(
                    'DSR_noSightOneHot', 'w/o sight ID')
        elif 'ucb(' in var_str_without_row_col:  # DSR
            if pc.change_ucb_name_to_dsr:
                var_str_without_row_col = f'{comb["alg"].upper()} + DSR (ours)'
            elif pc.change_ucb_name_by_mask_assuming_only_mask_both:
                if not pc.change_ucb_name_by_mask_assuming_only_mask_both_not_showing_with_or_without:
                    # 概念上只有 MaskBoth，但名字上若 originalState 則應實際是 MaskObs 所以名字是 MaskObs
                    if comb['mask_where'] == 'MaskBoth':
                        suffix_str = 'w/o state'
                    elif comb['mask_where'] == 'MaskObs':
                        suffix_str = 'w/  state'
                    else:
                        raise ValueError(f'Unknown mask_where: {comb["mask_where"]}')
                    var_str_without_row_col = f'{comb["alg"].upper()} + DSR {suffix_str}'
                else:
                    var_str_without_row_col = f'{comb["alg"].upper()} + DSR (ours)'
            elif pc.change_ucb_name_to_ucb_with_sights:
                var_str_without_row_col = var_str_without_row_col.replace(
                    'ucb(', 'UCB: [').replace(')', ']')
            else:
                pass
        elif 'StateLeaveObsNumber' in var_str_without_row_col:
            n_ag_obs = var_str_without_row_col[len('StateLeaveObsNumber'):]
            var_str_without_row_col = f'Number of agent obs. = {n_ag_obs}'
        elif 'equal(' in var_str_without_row_col:
            if 'equal(2,4,6,8,10)~5e6' in var_str_without_row_col:
                var_str_without_row_col = var_str_without_row_col.replace(
                    'equal(2,4,6,8,10)~5e6',
                    # 'Schedule: 2$\\rightarrow$4$\\rightarrow$6$\\rightarrow$8$\\rightarrow$10 each 1M')
                    'QMIX + Scheduling')
        elif 'original_' in var_str_without_row_col:
            # 若是 "original_???s"
            if (var_str_without_row_col.startswith('original_') and var_str_without_row_col[-1] == 's' and
                    var_str_without_row_col[9:-1].isdigit()):
                sight_value = var_str_without_row_col[9:-1]
                # var_str_without_row_col = f'Original ({sight_value}s)'
                # 10/17
                var_str_without_row_col = f'QMIX $d$={sight_value}'
        else:
            raise ValueError(f'Unknown var_str_without_row_col: {var_str_without_row_col}')

        if use_masia:
            var_str_without_row_col += ' + MASIA'
            var_str_without_row_col = var_str_without_row_col.replace('DSR (ours) + MASIA', 'MASIA + DSR (ours)')

        if config['domain'] == 'SMAC':
            # Replace MaskNone with ""
            var_str_without_row_col = var_str_without_row_col.replace('MaskNone', '')
            var_str_without_row_col = var_str_without_row_col.replace('MaskNon', '')

        # =========================================================================

        hp_constraints = comb_to_hp_constraints[comb]
        try:
            env_id = get_env_id(config, hp_constraints)
        except Exception as e:
            print(f"Error when getting env_id for {comb}. Config:")
            pprint(config)
            raise e

        i_row, i_col = row_var_values.index(row_var_value), col_var_values.index(col_var_value)

        # Get the subplot axis
        if n_rows == 1 and n_cols == 1:
            ax = axs
        elif n_cols == 1:
            ax = axs[i_row]
        elif n_rows == 1:
            ax = axs[i_col]
        else:
            ax = axs[i_row, i_col]

        runs_path = os.path.join(pc.f, method_name, env_id)
        sight_x = []
        sight_y = []

        # # If non-existing, raise an error
        # if not os.path.exists(runs_path):
        #     raise ValueError(f'{runs_path} does not exist (comb: {comb})')
        # If non-existing, skip
        if not os.path.exists(runs_path):
            p_bar.update(1)
            continue

        for exp_dir in os.listdir(runs_path):
            exp_dir_path = os.path.join(runs_path, exp_dir)
            if exp_dir.startswith('_') or not os.path.isdir(exp_dir_path):
                continue  # Skip the non-dir files and the hidden files

            assert METRICS_JSON_NAME in os.listdir(exp_dir_path), f'{exp_dir} does not have {METRICS_JSON_NAME}'
            metric_path = os.path.join(exp_dir_path, METRICS_JSON_NAME)
            metric_data = json.load(open(metric_path, 'r'))

            try:
                test_return_step_to_mean = {step: mean for step, mean in
                                            zip(metric_data[pc.tag]['steps'],
                                                metric_data[pc.tag]['values'])}
            except KeyError:
                warnings.warn(f'{metric_path} does not have the tag {pc.tag}')
                continue

            sight_x.append(list(test_return_step_to_mean.keys()))
            sight_y.append(list(test_return_step_to_mean.values()))

        # assert len(sight_x) > 0, f'No data found in {runs_path}'
        if len(sight_x) == 0:
            p_bar.update(1)
            continue

        # Set the number of data limit
        if isinstance(pc.n_tag_data, str):
            if pc.n_tag_data == 'auto':
                if pc.tag == 'test_return_mean':
                    n_tag_data = config['hp_alg'][comb['alg']]['min_n_tests']
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

        # Output to a file
        if pc.output_each_run_data:
            # Use comb to form a file name
            # Write into f'output_each_run_data/{comb}.npz'
            # If exists, overwrite
            output_dir = 'output_each_run_data'
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'{comb}.npz')
            np.savez(output_path, sight_y=sight_y, sight_x=sight_x)

        # Compute avg and ci
        sight_y_mean = get_mean_func(pc, axis=0)(np.array(sight_y))
        sight_x_mean = get_mean_func(pc, axis=0)(np.array(sight_x))

        # Compute the mean of all seeds' max
        sight_each_seed_max = [max(y) for y in sight_y]
        sight_each_seed_max_mean = get_mean_func(pc, axis=None)(sight_each_seed_max)

        # Compute mean of all seeds' last
        sight_each_seed_last = [y[-1] for y in sight_y]
        sight_each_seed_last_mean = get_mean_func(pc, axis=None)(sight_each_seed_last)

        # Compute mean of all seeds' mean of all testing points
        sight_each_seed_curve_mean = [np.mean(y) for y in sight_y]
        sight_each_seed_curve_mean_mean = get_mean_func(pc, axis=0)(sight_each_seed_curve_mean)
        # Compute mean of all seeds' mean of all testing points (best 50%)
        sight_each_seed_curve_mean_best_half = [np.mean(sorted(y)[-len(y) // 2:]) for y in sight_y]
        sight_each_seed_curve_mean_best_half_mean = get_mean_func(pc, axis=0)(sight_each_seed_curve_mean_best_half)

        # Compute 整體性
        # If only one run -> bandwidth is set to 0 manually
        # Init using mean
        if sight_n_runs <= 1:
            sight_y_upp = sight_y_mean
            sight_y_low = sight_y_mean
            sight_each_seed_max_upp = sight_each_seed_max_mean
            # noinspection PyUnusedLocal
            sight_each_seed_max_low = sight_each_seed_max_mean
            sight_each_seed_last_upp = sight_each_seed_last_mean
            # noinspection PyUnusedLocal
            sight_each_seed_last_low = sight_each_seed_last_mean
            # noinspection PyUnusedLocal
            sight_each_seed_curve_mean_upp = sight_each_seed_curve_mean_mean
            # noinspection PyUnusedLocal
            sight_each_seed_curve_mean_low = sight_each_seed_curve_mean_mean
            # noinspection PyUnusedLocal
            sight_each_seed_curve_mean_best_half_upp = sight_each_seed_curve_mean_best_half_mean
        else:
            if pc.band == 'std':
                sight_y_upp = sight_y_mean + np.std(sight_y, axis=0)
                sight_y_low = sight_y_mean - np.std(sight_y, axis=0)
                sight_each_seed_max_upp = sight_each_seed_max_mean + np.std(sight_each_seed_max)
                sight_each_seed_last_upp = sight_each_seed_last_mean + np.std(sight_each_seed_last)
                sight_each_seed_curve_mean_upp = sight_each_seed_curve_mean_mean + np.std(sight_each_seed_curve_mean)
                sight_each_seed_curve_mean_best_half_upp = sight_each_seed_curve_mean_best_half_mean + np.std(
                    sight_each_seed_curve_mean_best_half)
            elif pc.band == 'stderr':
                sem = stats.sem(sight_y, axis=0)
                sight_y_upp = sight_y_mean + sem
                sight_y_low = sight_y_mean - sem
                sight_each_seed_max_upp = sight_each_seed_max_mean + stats.sem(sight_each_seed_max)
                sight_each_seed_last_upp = sight_each_seed_last_mean + stats.sem(sight_each_seed_last)
                sight_each_seed_curve_mean_upp = sight_each_seed_curve_mean_mean + stats.sem(sight_each_seed_curve_mean)
                sight_each_seed_curve_mean_best_half_upp = sight_each_seed_curve_mean_best_half_mean + stats.sem(
                    sight_each_seed_curve_mean_best_half)
            elif pc.band == 'ci':
                (sight_y_low, sight_y_upp) = stats.t.interval(0.95, len(sight_y) - 1, loc=sight_y_mean,
                                                              scale=stats.sem(sight_y))
                (sight_each_seed_max_low, sight_each_seed_max_upp) = stats.t.interval(0.95, len(
                    sight_each_seed_max) - 1, loc=sight_each_seed_max_mean, scale=stats.sem(
                    sight_each_seed_max))
                (sight_each_seed_last_low, sight_each_seed_last_upp) = stats.t.interval(0.95, len(
                    sight_each_seed_last) - 1, loc=sight_each_seed_last_mean, scale=stats.sem(
                    sight_each_seed_last))
                (sight_each_seed_curve_mean_low, sight_each_seed_curve_mean_upp) = stats.t.interval(0.95, len(
                    sight_each_seed_curve_mean) - 1, loc=sight_each_seed_curve_mean_mean, scale=stats.sem(
                    sight_each_seed_curve_mean))
                (sight_each_seed_curve_mean_best_half_low, sight_each_seed_curve_mean_best_half_upp) = stats.t.interval(
                    0.95, len(sight_each_seed_curve_mean_best_half) - 1, loc=sight_each_seed_curve_mean_best_half_mean,
                    scale=stats.sem(sight_each_seed_curve_mean_best_half))
            elif pc.band == 'minmax':
                sight_y_upp = np.max(sight_y, axis=0)
                sight_y_low = np.min(sight_y, axis=0)
                sight_each_seed_max_upp = np.max(sight_each_seed_max)
                # noinspection PyUnusedLocal
                sight_each_seed_max_low = np.min(sight_each_seed_max)
                sight_each_seed_last_upp = np.max(sight_each_seed_last)
                # noinspection PyUnusedLocal
                sight_each_seed_last_low = np.min(sight_each_seed_last)
                sight_each_seed_curve_mean_upp = np.max(sight_each_seed_curve_mean)
                sight_each_seed_curve_mean_best_half_upp = np.max(sight_each_seed_curve_mean_best_half)
            else:
                raise ValueError(f'Invalid band: {pc.band}')

        # =============== Plot a line ===============

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
        label = f'{var_str_without_row_col:12}'
        plot_label_details = not pc.label_without_len_and_runs or not pc.label_without_overall_performance
        if plot_label_details:
            label += ' ('
        if not pc.label_without_len_and_runs:
            label += f'len:{len(sight_x_plotted)},runs:{sight_n_runs}'
        if not pc.label_without_overall_performance:
            label += (
                # --- 解註解以顯示 mean of meax ---

                # f',max:{sight_each_seed_max_mean:.3f}'
                # f'±{sight_each_seed_max_upp - sight_each_seed_max_mean:.3f},'

                # --- 解註解以顯示 mean of last ---

                f'last:{sight_each_seed_last_mean:.3f}'
                f'±{sight_each_seed_last_upp - sight_each_seed_last_mean:.3f},'

                # --- 解註解以顯示 mean of curve mean ---

                # f'curve:{sight_each_seed_curve_mean_mean:.3f}'
                # f'±{sight_each_seed_curve_mean_upp - sight_each_seed_curve_mean_mean:.3f},'

                # --- 解註解以顯示 mean of best 1/2 curve mean ---

                # f'best1/2Mean:{sight_each_seed_curve_mean_best_half_mean:.3f}'
                # f'±{sight_each_seed_curve_mean_best_half_upp - sight_each_seed_curve_mean_best_half_mean:.3f})'
            )

        # Gather the last mean and std
        # Check if existing first
        if method_name in row_var_to_col_var_to_value_dict[row_var_value][col_var_value]:
            raise ValueError(f'Already have {method_name} in {row_var_value} {col_var_value}')
        row_var_to_col_var_to_value_dict[row_var_value][col_var_value][method_name] = (
            sight_each_seed_last_mean, sight_each_seed_last_upp - sight_each_seed_last_mean)

        if plot_label_details:
            label += ')'

        # Plot kwargs
        plot_kwargs: defaultdict[str, Any] = defaultdict()
        plot_kwargs['linewidth'] = pc.linewidth

        # Default colors (淺到深)
        if pc.pure_sights or config['domain'] in ['LBF', 'RWARE']:

            # Get the idx of this side in the pc.sights if given
            pure_sights = pc.pure_sights
            color_idx = None
            if pc.pure_sights and comb['sight'] in pc.pure_sights:
                color_idx = pc.pure_sights.index(comb['sight'])

            # --- For some domains, override the color ---
            comb_sight = comb['sight']
            comb_map = comb['map']
            if config['domain'] == 'LBF':
                sight_list_6s = ['2s', '4s', '6s']
                sight_list_10s = ['2s', '4s', '6s', '8s', '10s']
                sight_list_15s = ['3s', '6s', '9s', '12s', '15s']
                # print(f'comb_map: {comb_map}, comb_sight: {comb_sight}')
                if '6s-' in comb_map and comb_sight in sight_list_6s:
                    pure_sights = sight_list_6s
                    color_idx = sight_list_6s.index(comb_sight)
                elif ('10s-' in comb_map and comb_sight in sight_list_10s) or (
                        # any([',10)' in comb[var_] for var_ in comb.keys()]) or
                        comb_sight == 'original_10s'
                ):
                    if comb_sight == 'original_10s':
                        comb_sight = '10s'
                    # elif:
                    #     color_idx = None
                    else:
                        color_idx = sight_list_10s.index(comb_sight)

                    pure_sights = sight_list_10s
                    color_idx = sight_list_10s.index(comb_sight)
                elif '15s-' in comb_map and comb_sight in sight_list_15s:
                    pure_sights = sight_list_15s
                    color_idx = sight_list_15s.index(comb_sight)
            elif config['domain'] == 'RWARE':
                sight_list_3s = ['1s', '2s', '3s']
                sight_list_5s = ['1s', '2s', '3s', '4s', '5s']
                if '-3s' in comb_map and comb_sight in sight_list_3s:
                    pure_sights = sight_list_3s
                    color_idx = sight_list_3s.index(comb_sight)
                elif '-5s' in comb_map and comb_sight in sight_list_5s:
                    pure_sights = sight_list_5s
                    color_idx = sight_list_5s.index(comb_sight)

            if pure_sights:
                if len(pure_sights) <= 4:
                    BLUE_START, BLUE_END = 0.5, 1
                else:
                    BLUE_START, BLUE_END = 0.4, 1
                if 'use_rnn' in comb.keys() and comb['use_rnn'] == 'MLP':
                    cool_colors = plt.cm.Greens(np.linspace(BLUE_START, BLUE_END, len(pure_sights)))
                else:  # Normal cases
                    cool_colors = plt.cm.Blues(np.linspace(BLUE_START, BLUE_END, len(pure_sights)))

                plot_kwargs['color'] = cool_colors[color_idx] if color_idx is not None else None
            else:
                plot_kwargs['color'] = None

        z_order = None
        # If 'ucb' or 'DSR' in any of the comb.values(), move to the front
        if any(['ucb' in exp_var_value for exp_var_value in comb.values()]) or any(
                ['DSR' in exp_var_value for exp_var_value in comb.values()]):
            z_order = 10
        plot_kwargs['zorder'] = z_order

        # Check custom style
        n_satisfied = 0
        for style_dict in pc.list_of_assigned_styles_and_keywords:
            keywords = style_dict['keywords']
            # 2024025 version: only match substring ✓
            satisfied = True
            for keyword in keywords:
                keyword_match = any([keyword in exp_var_value for exp_var_value in comb.values()])
                if not keyword_match:
                    satisfied = False
                    break

            if satisfied:
                # If the keywords are all in the alg map_name sight, apply the kwargs
                plot_kwargs.update({k: v for k, v in style_dict.items() if k != 'keywords'})
                n_satisfied += 1

        # Plot
        line_ret, = ax.plot(sight_x_plotted, sight_y_plotted, label=label, **plot_kwargs)

        # Fill between
        fill_between_kwargs = dict(alpha=0.1)
        # if 'color' in plot_kwargs and plot_kwargs['color'] is not None:
        #     fill_between_kwargs['color'] = plot_kwargs['color']
        # else:
        #     fill_between_kwargs['color'] = line_ret.get_color()
        fill_between_kwargs['color'] = line_ret.get_color()
        ax.fill_between(sight_x_plotted, sight_band_low, sight_band_upp, **fill_between_kwargs)

        # If alg str too long, we need to cut it and add ...
        var_str_for_text = var_str_without_row_col if len(var_str_without_row_col) <= 12 else var_str_without_row_col[
                                                                                              :12] + '...'

        # Text: at the end of the line and other positions
        if pc.show_text_on_line:
            ax.text(sight_x_plotted[-1], sight_y_plotted[-1], var_str_for_text, fontsize=9, color='black',
                    verticalalignment='center')
            plot_text_positions = []  # e.g., [2, 5] (十分之幾)
            for plot_text_position in plot_text_positions:
                ax.text(sight_x_plotted[int(len(sight_x_plotted) / 10 * plot_text_position)],
                        sight_y_plotted[int(len(sight_y_plotted) / 10 * plot_text_position)],
                        var_str_for_text, fontsize=6, color='black', verticalalignment='center')

        if pc.show_title:
            title_str = f'{var_str_for_text}'
            ax.set_title(title_str, fontdict=GLOBAL_FONT_TITLE)
        if pc.show_legend:
            # Get handles and labels
            handles, labels = ax.get_legend_handles_labels()
            if pc.sort_legend_by_suffix_digits:
                # Sort labels and handles by suffix digits
                sorted_labels_handles = sorted(zip(labels, handles), key=lambda x: extract_number_from_label(x[0]))
                sorted_labels, sorted_handles = zip(*sorted_labels_handles)
            elif pc.sort_legend_by_alphabet:
                sorted_labels_handles = sorted(zip(labels, handles), key=lambda x: x[0])
                sorted_labels, sorted_handles = zip(*sorted_labels_handles)
            else:
                # Sort labels and handles (1) alphabetically (2) length
                sorted_labels_handles = sorted(zip(labels, handles), key=lambda x: x[0])  # Sort by label alphabetically
                sorted_labels_handles = sorted(sorted_labels_handles, key=lambda x: len(x[0]))
                sorted_labels, sorted_handles = zip(*sorted_labels_handles)
            # Create sorted legend
            legend_kwargs = dict(loc=pc.legend_loc) if pc.legend_loc else dict()
            # noinspection PyTypeChecker
            legend_kwargs['fontsize'] = LEGEND_FONT_SIZE
            legend = ax.legend(sorted_handles, sorted_labels, **legend_kwargs)
            legend.set_zorder(LEGEND_ZORDER)

        ax.grid(True)
        ax.set_xlabel(pc.xlabel, fontdict=GLOBAL_FONT_XY_LABEL)
        ax.set_ylabel(pc.ylabel, fontdict=GLOBAL_FONT_XY_LABEL)

        # Tick size
        ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
        ax.xaxis.get_offset_text().set_fontsize(TICK_SIZE)  # 調整 x 軸的 offset label 字體大小
        ax.yaxis.get_offset_text().set_fontsize(TICK_SIZE)  # 調整 y 軸的 offset label 字體大小

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            # label.set_fontproperties(GLOBAL_FONT_SERIF)
            assert len(GLOBAL_FONT_SERIF) == 1, 'Only one font is supported'
            label.set_fontproperties(font_manager.FontProperties(family=GLOBAL_FONT_SERIF[0]))

        p_bar.update(1)

        # y scale
        ax.set_yscale(pc.y_scale)

        # Check custom style
        n_satisfied = 0
        for style_dict in pc.list_of_assigned_styles_and_keywords:
            keywords = style_dict['keywords']
            # 2024025 version: only match substring ✓
            satisfied = True
            for keyword in keywords:
                keyword_match = any([keyword in exp_var_value for exp_var_value in comb.values()])
                if not keyword_match:
                    satisfied = False
                    break

            if satisfied:
                # If the keywords are all in the alg map_name sight, apply the kwargs
                plot_kwargs.update({k: v for k, v in style_dict.items() if k != 'keywords'})
                n_satisfied += 1

        x_low = pc.x_low
        x_high = pc.x_high
        y_low = pc.y_low
        y_high = pc.y_high
        if pc.map_keyword_to_low_or_high or pc.sight_keyword_to_low_or_high:
            targets = None
            items = None
            if pc.map_keyword_to_low_or_high:
                targets = comb['map']
                items = pc.map_keyword_to_low_or_high.items()
            elif pc.sight_keyword_to_low_or_high:
                targets = comb['sight']
                items = pc.sight_keyword_to_low_or_high.items()
            else:
                raise ValueError('Unknown target')
            for keyword, low_high_dict in items:
                if keyword in targets:
                    for low_high_key, low_high_value in low_high_dict.items():
                        if low_high_key == 'y_low':
                            y_low = float(low_high_value)
                        elif low_high_key == 'y_high':
                            y_high = float(low_high_value)
                        elif low_high_key == 'x_low':
                            x_low = float(low_high_value)
                        elif low_high_key == 'x_high':
                            x_high = float(low_high_value)
                        else:
                            raise ValueError(f'Unknown key: {low_high_key}')
        ax.set_xlim(x_low, x_high)
        ax.set_ylim(y_low, y_high)

    p_bar.close()

    # Title
    if pc.title:
        if not pc.hide_subtitle_description:
            only_one_uniq_var_str = ",".join(
                [f'{k}={v}' for k, v in only_one_unique_value_var_and_values.items() if
                 k not in [row_var, col_var]])
            title_desc = f'\n(tag={pc.tag}) (mean±{BAND_NAMES[pc.band]}) ({only_one_uniq_var_str})'
        else:
            title_desc = ''
        if pc.show_subtitle:
            subtitle = f'{pc.title}{title_desc}'
            fig.suptitle(subtitle, fontsize=30)

    # ========================= Output a Table ========================================

    # =============== 替换为保存CSV文件的部分 =================
    import csv
    metric_table_save_dir = 'exp_plot/table/'
    os.makedirs(metric_table_save_dir, exist_ok=True)
    csv_save_path = os.path.join(metric_table_save_dir, f'{pc.title}.csv')
    with open(csv_save_path, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';')  # 使用';'作为分隔符
        writer.writerow(["row_var_value", "col_var_value", "method_name", "mean+-std"])  # CSV头

        # 将mean和std格式化为小数点后三位，并输出"mean+-std"格式
        for row_var_value, col_var_to_value_dict in row_var_to_col_var_to_value_dict.items():
            for col_var_value, value_dict in col_var_to_value_dict.items():
                for method_name, (mean, std) in value_dict.items():
                    formatted_result = f'{mean:.3f} ± {std:.3f}'
                    writer.writerow([row_var_value, col_var_value, method_name, formatted_result])

    print(f'Saved summary table to {csv_save_path}')

    # =================================================================

    # plt.tight_layout()
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    img_save_path = pc.save_path
    if img_save_path[-1] == '/':
        # img_save_path += f'{pc.title}.png'
        img_save_path += f'{pc.title}.pdf'
    img_save_path = os.path.join(args.save, img_save_path)

    # Create the dir if not exists
    os.makedirs(os.path.dirname(img_save_path), exist_ok=True)

    plt.savefig(img_save_path)
    print(f'Saved to {img_save_path}')
    if not pc.no_show:
        plt.show()
    plt.close()
    print('Done!')


def check_whether_this_yml_first_level_key_is_valid(config: dict, yml_path: Optional[str] = None):
    """檢查一個 yaml 檔案的第一層 key 是否合法。"""
    path_str = f'(path: {yml_path})' if yml_path is not None else ''
    # 檢查每個 first level key。如果有不認識的 key，就會報錯
    for key in config.keys():
        if key not in MUST_HAVE_KEYS + OPTIONAL_KEYS:
            raise ValueError(f"Unknown key \"{key}\" in the yaml file {path_str}")
    # 檢查必要的 key 是否都有
    for key in MUST_HAVE_KEYS:
        if key not in config:
            raise ValueError(f"Key \"{key}\" is missing in the yaml file {path_str}")


def recursive_update_dict(d, u, overwrite=True):
    """給定兩個 dict，將 u 的值更新到 d 中，如果 d 中不存在該 leaf key 的話。"""
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = recursive_update_dict(d.get(k, {}), v, overwrite=overwrite)
        else:
            if overwrite:
                d[k] = v
            else:
                if k not in d:  # 如果 d 中不存在這個 leaf key，就更新
                    d[k] = v
    return d


def get_comb_hyperparam_constraints(config: dict, comb: ExpVarComb):
    """從 config 去讀這個 comb 對應的超參設定。"""
    # hp_all
    hp_constraints = deepcopy(config['hp_all'])
    # hp_alg_group
    alg_group = config['hp_alg'][comb['alg']]['hyperparameter_group']
    if alg_group is not None:
        hp_constraints = recursive_update_dict(hp_constraints, config['hp_alg_group'][alg_group])
    # hp_alg
    hp_constraints = recursive_update_dict(hp_constraints, config['hp_alg'][comb['alg']]['custom_hyperparameters'])
    # hp_map
    if config.get('hp_map', None) is not None:
        hp_constraints = recursive_update_dict(hp_constraints, config['hp_map'][comb['map']])
    # hp_sight
    if config.get('hp_sight', None) is not None:
        hp_constraints = recursive_update_dict(hp_constraints, config['hp_sight'][comb['sight']])
    # hp_map_sight
    if config.get('hp_map_sight', None) is not None:
        # print('~~~~~~~~~~~', comb)
        if comb["map"] in config['hp_map_sight'] and comb["sight"] in config['hp_map_sight'][comb["map"]]:
            hp_constraints = recursive_update_dict(hp_constraints, config['hp_map_sight'][comb["map"]][comb["sight"]])
            # print(comb, hp_constraints['env_args'])
    if not (config.get('hp_map_sight', None) is not None) and not (config.get('hp_sight', None) is not None) and not (
            config.get('hp_map', None) is not None):
        warnings.warn('No hp_map_sight, hp_sight, hp_map in the config yaml')
    # hp_other_variables
    for other_var, other_var_value in comb.params:
        if other_var in config['hp_other_variables']:
            hp_constraints = recursive_update_dict(hp_constraints,
                                                   config['hp_other_variables'][other_var][other_var_value])
        else:
            if other_var not in ['alg', 'map', 'sight']:
                raise ValueError(f'Unknown other variable: {other_var}')
    return hp_constraints


class Colors:  # ANSI escape codes
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'  # 用來重置顏色


def check_run_valid(seed, run: dict, each_info: bool = True, simplified_path: bool = False,
                    force_show_info=False) -> [str, bool]:
    """檢查 seed 的 valid 狀態，並回傳訊息"""
    msg = ''
    if run['path']:
        path_str = f'path={run["path"]} ' if not simplified_path else f'path={run["path"].split("/")[-1]}'
    else:
        path_str = ''
    tmp_msg = ''
    if not run['valid_test_interval']:
        tmp_msg += f'test_interval not valid'
    if not run['valid_test_return_mean_numbers']:
        if len(tmp_msg) > 0:
            tmp_msg += ', '
        tmp_msg += f'{Colors.YELLOW}test_return_mean_numbers not valid{Colors.RESET}'
    if len(tmp_msg) > 0 or force_show_info:  # Invalid
        msg = f'{path_str} seed={seed} {tmp_msg}'
        if each_info:
            last_step = run['test_steps'][-1] if run['test_steps'] else 'None'
            last_timestamp = run['test_timestamps'][-1] if run['test_timestamps'] else 'None'
            msg += (f' (n_tests={len(run["test_steps"])}, last_step={last_step}, '
                    f'last_time={Colors.RED}{last_timestamp}{Colors.RESET})')
        msg += '\n'
    return msg, len(tmp_msg) == 0


def check_leaves_in_nested_dict(dict_a, dict_b, problems: list[str], a_name=None, b_name=None,
                                not_having_ok_hyperparameters: Optional[list] = None) -> None:
    """
    檢查 A 的 leaf 是否都在 B 的 leaf list 中。
    給定2個巢狀dict A 和 B，B 的 leaf 都是 list of elements (這些元素為可接受值)。檢查 A 的 leaf 是否都在 B 的 leaf list。

    A: dict，巢狀結構，其 leaf 是值。
    B: dict，巢狀結構，其 leaf 是 list。
    """
    if not isinstance(dict_a, dict) or not isinstance(dict_b, dict):
        raise ValueError('dict_a and dict_b should be dict type.')
    for key, value in dict_a.items():
        if key not in dict_b:
            # Check 看有沒有在 not_having_ok_hyperparameters 中，若有關鍵字符合的話就skip
            if not_having_ok_hyperparameters:
                skip = any([nh in key for nh in not_having_ok_hyperparameters])
                if skip:
                    print(f'key={key}, skip={skip}')
                    continue
            else:
                problems.append(f'{a_name}.{key} not found in {b_name}')
        else:
            if isinstance(value, dict):
                # 如果當前值是字典，遞歸檢查
                check_leaves_in_nested_dict(value, dict_b[key], a_name=f'{a_name}.{key}', b_name=f'{b_name}.{key}',
                                            problems=problems,
                                            not_having_ok_hyperparameters=not_having_ok_hyperparameters)
            else:
                # 如果當前值不是字典，檢查它是否在 B 的對應 leaf list 中
                try:
                    if value not in dict_b[key]:
                        problems.append(f'{a_name}.{key}={value} not found in {b_name}.{key}={dict_b[key]}')
                except TypeError as e:
                    print(f'Error when checking {a_name}.{key}={value} in {b_name}.{key}={dict_b[key]}'
                          f' (value type: {type(value)}, dict_b[key] type: {type(dict_b[key])})')
                    raise e


# noinspection DuplicatedCode
def check_exp(config: dict, args, all_combinations: list[ExpVarComb], all_exp_vars: list[str]):
    # 一個一個 comb 檢查
    col_alg = []
    col_sight = []
    col_map = []
    col_other_vars: list[tuple] = []
    col_msg = []
    col_env_id = []
    col_method_names = []
    all_done = True
    n_done = 0
    n_not_yet = 0
    comb_to_hp_constraints = {}
    for comb in tqdm(all_combinations, desc='Checking exp var combinations', disable=args.no_check):
        if set(comb.keys()) != set(all_exp_vars):
            raise ValueError(f'exp_vars should be {all_exp_vars}, but got {comb.keys()}')
        # Get this exp comb's hp constraints
        hp_constraints = get_comb_hyperparam_constraints(config, comb)
        comb_to_hp_constraints[comb] = hp_constraints

        if args.no_check:
            continue

        # Get method name (shown in sacred)
        alg = comb['alg']
        map_name = comb['map']
        sight = comb['sight']
        method_name = comb.get_method_name(all_exp_vars)

        # Get env_id
        try:
            env_id = get_env_id(config, hp_constraints)
        except Exception as e:
            print(f"Error getting env_id for {comb}, and it's hp_constraints:")
            pprint(hp_constraints)
            print(f'Config:')
            pprint(config)
            raise e

        # Get the path of this comb runs
        runs_dir = str(os.path.join(args.folder, method_name, env_id))
        # Get run names and sort them
        if not os.path.exists(runs_dir):
            print(f'comb={comb} not found in {runs_dir}')

        run_names = [d for d in os.listdir(runs_dir) if
                     os.path.isdir(os.path.join(runs_dir, d)) and not d.startswith('_')]
        run_names = sorted(run_names, key=lambda x: (len(x), x))
        if len(run_names) == 0:
            warnings.warn(f'Empty run names for {method_name}/{env_id}')
            if args.delete:
                os.rmdir(runs_dir)
                print(f'Deleted empty folder: {runs_dir}')

        # Check each run
        seed_to_runs = defaultdict(list)
        for run_name in run_names:
            # Load config and metric data
            run_dir = os.path.join(runs_dir, run_name)
            config_path = os.path.join(run_dir, 'config.json')
            if not os.path.exists(config_path):
                raise FileNotFoundError(f'config.json not found in {run_dir}')
            metric_path = os.path.join(run_dir, METRICS_JSON_NAME)
            if not os.path.exists(metric_path):
                raise FileNotFoundError(f'{METRICS_JSON_NAME} not found in {run_dir}')

            error_msg = None
            config_data = None
            metric_data = None
            try:
                config_data = json.load(open(config_path, 'r'))
                try:
                    metric_data = json.load(open(metric_path, 'r'))
                except Exception as e:
                    error_msg = f'Error loading {METRICS_JSON_NAME} in {run_dir}: {e}'
                    # raise e
            except Exception as e:
                error_msg = f'Error loading config.json in {run_dir}: {e}'
                # raise e

            # Start to collect problems of this run
            hp_problems: list[str] = []  # a list of problem messages

            if error_msg is not None:
                hp_problems.append(error_msg)
                warnings.warn(error_msg)
            else:
                # Get data from config.json
                seed = config_data['seed']
                test_interval_configured = config_data['test_interval']

                # Get test return data
                metric_test_return_dict = metric_data.get('test_return_mean', None)
                if metric_test_return_dict is None:
                    hp_problems.append(f'test_return_mean not found in {METRICS_JSON_NAME}.')
                    metric_test_return_dict = {'values': [], 'steps': []}
                test_return_means = metric_test_return_dict.get('values', None)
                if test_return_means is None:
                    hp_problems.append(f'values not found in {METRICS_JSON_NAME} test_return_mean dict.')
                    test_return_means = []
                test_steps = metric_test_return_dict.get('steps', None)
                if test_steps is None:
                    hp_problems.append(f'steps not found in {METRICS_JSON_NAME} test_return_mean dict.')
                    test_steps = []
                    test_timestamps = []
                else:
                    test_timestamps = metric_test_return_dict.get('timestamps', None)

                # Whether test_interval and n tests valid
                assert len(hp_constraints['test_interval']) == 1, 'test_interval should only have one value.'
                valid_configured_test_interval = test_interval_configured == hp_constraints['test_interval'][0]
                valid_empirical_test_return_numbers = len(test_return_means) >= config['hp_alg'][alg]['min_n_tests']

                # Check if all hp loaded from config.json meet our requirements; if not, list all the problems
                # Take out the keys that are in hp_all_skip first
                dict_a = {k: v for k, v in config_data.items() if k not in config['hp_all_skip']}
                dict_b = {k: v for k, v in hp_constraints.items() if k not in config['hp_all_skip']}
                check_leaves_in_nested_dict(
                    dict_a, dict_b, hp_problems, a_name='config', b_name='hp_constraints',
                    not_having_ok_hyperparameters=config['hp_alg'][alg]['not_having_ok_hyperparameters'])

                # 這裡反過來看有yml config的超參是否都有在run的config.json裡面
                for hp_name in hp_constraints.keys():
                    if hp_name not in config_data.keys():
                        if hp_name not in config['hp_alg'][alg]['not_having_ok_hyperparameters']:
                            raise ValueError(f'Unknown hyperparameter: {hp_name} in config.yaml')

                # Construct a RUN object
                run = dict(path=run_dir,
                           seed=seed,
                           test_interval=test_interval_configured,
                           test_return_means=test_return_means,
                           valid_test_interval=valid_configured_test_interval,
                           valid_test_return_mean_numbers=valid_empirical_test_return_numbers,
                           alg_name=alg,
                           sight_name=sight,
                           map_name=map_name,
                           method_name=method_name,
                           config=config_data,
                           hp_problems=hp_problems,
                           test_steps=test_steps,
                           test_timestamps=test_timestamps)
                seed_to_runs[seed].append(run)

            # Automatically delete invalid runs
            if args.delete:
                delete_hp_problems = hp_problems.copy()
                if not valid_empirical_test_return_numbers:
                    delete_hp_problems.append(f'Invalid test return numbers. Wanted: '
                                              f'{config["hp_alg"][alg]["min_n_tests"]} '
                                              f'but got: {len(test_return_means)} '
                                              f'(last_step={test_steps[-1] if test_steps else "None"})')
                if not valid_configured_test_interval:
                    delete_hp_problems.append(f'Invalid test interval. Wanted: {hp_constraints["test_interval"]} '
                                              f'but got: {test_interval_configured}')
                if len(delete_hp_problems) > 0:
                    problem_str = "\n".join(delete_hp_problems)
                    print(
                        f'[Invalid Run] comb: {comb}\n'
                        f'seed={seed} path={run_dir}\n'
                        f'Problems: {problem_str}', flush=True)
                    response = input('Delete this invalid run? (y/n): ')
                    if response == 'y':
                        try:
                            import shutil
                            shutil.rmtree(run_dir)
                            print(f'Deleted invalid run: {run_dir}')
                            hp_problems.append(f'Invalid run DELETED')
                        except Exception as e:
                            print(f'Error deleting invalid run: {run_dir}')
                            print(e)
                    else:
                        print(f'Not deleting invalid run: {run_dir}')
                        print()

        # Check duplicated/missing seeds

        # 註：若不想固定seed，只要求runs的數量，在config請設定`seeds` 為 integer。
        #     若要要求固定seed，請設定`seeds` 為 list of integers。

        # ---------- Make message ----------

        msg = ''

        fixed_seeds: bool = not isinstance(config['seeds'], int)
        n_seeds = len(config['seeds']) if fixed_seeds else config['seeds']

        if fixed_seeds:
            loop_seeds = config['seeds']
            # Check uniqueness
            if len(loop_seeds) != len(set(loop_seeds)):
                raise ValueError(f'Seeds should be unique: {loop_seeds}')
        else:
            # 直接用現有的 seeds。然後要確認總數是否正確。
            loop_seeds = list(seed_to_runs.keys())

        invalid_runs: list[dict] = []
        invalid_run_messages: list[str] = []
        not_finished_seeds = set(config['seeds'])

        for seed in loop_seeds:
            seed_runs = seed_to_runs[seed]

            # Add hp problem msg
            for run in seed_runs:
                if len(run["hp_problems"]) > 0:
                    hp_problem_total_str = "\n".join(run["hp_problems"])
                    run_msg = hp_problem_total_str
                    invalid_runs.append(run)
                    invalid_run_messages.append(run_msg)
                    msg += f'{seed} path={run["path"]} hp_problems: \n{hp_problem_total_str}; \n'

            # Check runs problems
            if len(seed_runs) == 0:
                if fixed_seeds:
                    msg += f'{Colors.CYAN}Seed={seed} not found{Colors.RESET}; \n'

            elif len(seed_runs) == 1:
                run_smg, is_valid = check_run_valid(seed, seed_runs[0], each_info=True, simplified_path=True,
                                                    force_show_info=False)
                if not is_valid:
                    invalid_runs.append(seed_runs[0])
                    invalid_run_messages.append(run_smg)
                else:
                    if seed in not_finished_seeds:
                        not_finished_seeds.remove(seed)
                msg += run_smg
            elif len(seed_runs) > 1:
                # Check each duplicated runs' path and valid status
                tmp_msg = ''
                for run in seed_runs:
                    run_msg, is_valid = check_run_valid(seed, run, each_info=True, simplified_path=True,
                                                        force_show_info=True)
                    if not is_valid:
                        invalid_runs.append(run)
                        invalid_run_messages.append(run_msg)
                    else:
                        if seed in not_finished_seeds:
                            not_finished_seeds.remove(seed)
                    tmp_msg += run_msg
                if fixed_seeds:
                    duplicated_desc_end = ": \n" if len(tmp_msg) > 0 else "."
                    msg += f'{Colors.BLUE}Seed={seed} duplicated{Colors.RESET}{duplicated_desc_end}'
                    msg += tmp_msg
                    msg += '\n'

        if not fixed_seeds and len(loop_seeds) > n_seeds:
            msg += f'{Colors.RED}Too many seeds: {len(loop_seeds)} > {n_seeds}{Colors.RESET}\n'

        # Fill the summary table, appending a row for this alg x sight combination
        if len(msg) == 0:
            msg = '✅'
            n_done += 1
        else:
            all_done = False
            n_not_yet += 1

        # Append to the lists
        col_alg.append(alg)
        col_sight.append(sight)
        col_map.append(map_name)
        col_other_vars.append(tuple([v for k, v in comb.params if k not in ['alg', 'map', 'sight']]))
        col_msg.append(msg)
        col_env_id.append(env_id)
        col_method_names.append(method_name)

    if args.no_check:
        print('Checking is skipped.')
    else:  # Output
        # ------------ 最後，畫一個 summary table，秀出哪些方法的哪些環境的哪些seed還沒有完成 ------------
        # Row0: alg name
        # Row1: sight name
        # Columns: map names
        # Cell: 該有的各seed完成了嗎？若沒完成是有什麼問題？(長度不足？有什麼valid的問題？重複seed？)

        # Create the DataFrame from the lists
        summary_table = pd.DataFrame({
            # 'alg': col_alg,
            # 'sight': col_sight,
            'dir': col_method_names,
            'map': col_map,
            'env_id': col_env_id,
            'msg': col_msg,
        })

        summary_table_to_print = summary_table.copy()
        summary_table_to_print['msg'] = summary_table_to_print['msg'].apply(lambda x: x.replace('\n', ' '))

        # # summary_table.loc[f'{alg}_{sight}', map_name] = msg
        # summary: dict[str, dict[str, dict[str, dict[int, str]]]] = {}  # alg -> sight -> map -> seed -> msg

        # Set df to print all the content
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        # Print the summary table
        print('========= Summary table =========')
        print(summary_table_to_print)

        # Also save as a csv file
        time_str = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        summary_table_dir = 'exp_summary_table_history'
        os.makedirs(summary_table_dir, exist_ok=True)
        summary_table.to_csv(os.path.join(summary_table_dir, 'check_exp_summary_table_' + time_str + '.csv'),
                             index=True)

        # Show successful message if all methods are done
        if all_done:
            print('★★★★★ All methods are done! ★★★★★')
        else:
            print('★★★★★ NOT all methods are done! ★★★★★')
        print('')
        print(f'Number of done: {n_done} runs, not yet: {n_not_yet} runs, total: {n_done + n_not_yet} runs.')
    return comb_to_hp_constraints


def read_all_config(args):
    config = {}
    yml_path = args.config
    while True:
        print(f'Loading {yml_path}')
        new_yml = load_yaml(yml_path)
        check_whether_this_yml_first_level_key_is_valid(new_yml, yml_path)
        config = recursive_update_dict(config, new_yml, overwrite=False)
        parent = new_yml['parent']
        if not (isinstance(parent, str) or parent is None):
            raise ValueError(f"parent must be a string or None, but got {parent}")
        if parent is None:
            break
        yml_path = parent
    return config


def read_plot_config(config: dict, args) -> tuple[dict[str, set[ExpVarComb]], dict[str, PlotConfig], set[ExpVarComb]]:
    """讀取畫圖的設定，看有哪些實驗組合是要畫的，綜合起來，我們來 check
    """
    assert 'plot' in config, 'No plot config found in the config yaml'
    assert 'plots' in config['plot'], 'No plots found in the plot config'

    # Load plotting templates
    template_name_to_dict = {}
    for template_dict in config['plot']['templates']:
        template_name = template_dict.pop('template_name')
        template_name_to_dict[template_name] = template_dict

    # Load each plot's settings
    each_plot_combinations = {}
    each_plot_pc = {}
    all_combinations = set()  # 所有畫圖統合起來全部要檢查的組合
    for plot in config['plot']['plots']:
        template_name = plot.pop('template_name')

        combinations = set()

        # Get template's exp vars
        template_exp_vars: dict = template_name_to_dict[template_name].get('exp_vars', {})
        template_skip_var_combinations: list = template_name_to_dict[template_name].get('skip_var_combinations', {})

        # Update exp_vars with this plot's custom settings
        exp_vars = plot.get('exp_vars', {})
        skip_var_combinations = plot.get('skip_var_combinations', [])
        exp_vars = recursive_update_dict(template_exp_vars, exp_vars, overwrite=True)
        skip_var_combinations += template_skip_var_combinations

        # Must have algs and maps
        for key in ['alg', 'map']:
            if key not in exp_vars:
                raise ValueError(f"Key \"{key}\" is missing in the plot's exp_vars")
        # Get all combinations
        for comb in product(*exp_vars.values()):
            exp_var_comb = ExpVarComb.get_exp_var_comb(dict(zip(exp_vars.keys(), comb)))
            combinations.add(exp_var_comb)

        # Get all combinations needing to skip
        for skip_comb_setting in skip_var_combinations:
            if skip_comb_setting.keys() != set(exp_vars.keys()):
                raise ValueError(f'skip_comb_setting should have the same keys as exp_vars,'
                                 f'but got {skip_comb_setting.keys()} vs {exp_vars.keys()}')
            for comb in product(*skip_comb_setting.values()):
                exp_var_comb = ExpVarComb.get_exp_var_comb(dict(zip(skip_comb_setting.keys(), comb)))
                combinations.discard(exp_var_comb)

        # Dynamic
        if 'save_path' in plot['plot_config']:
            save_path = plot['plot_config']['save_path']
        else:
            save_path = template_name_to_dict[template_name]['plot_config']['save_path']

        if save_path[-1] == '/':
            if 'title' in plot['plot_config']:
                title = plot['plot_config']['title']
            else:
                title = template_name_to_dict[template_name]['plot_config']['title']
            save_path += title + '.png'

        plot_name = save_path
        each_plot_combinations[plot_name] = combinations
        all_combinations.update(combinations)

        # 每一張圖的畫圖設置都是一個 PlotConfig 物件。

        # Init with the template's settings
        pc = PlotConfig(**template_name_to_dict[template_name]['plot_config'])
        pc.f = args.folder

        # Update with this plot's custom settings (overwrite the template's settings)
        for k, v in plot['plot_config'].items():
            setattr(pc, k, v)
        each_plot_pc[plot_name] = pc

        if args.verbose:
            print(f'Plot {plot_name}:')
            pprint(combinations)
            print()

    return each_plot_combinations, each_plot_pc, all_combinations


def main():
    args = get_args()

    # Read config from yaml recursively
    config = read_all_config(args)

    # All exp setting
    all_exp_vars: list[str] = ['alg', 'map', 'sight'] + list(config['hp_other_variables'].keys())

    # ============== 讀畫圖的設定 ==============

    if config['plot'] is None:
        raise ValueError("The plot key is missing in the yaml file")
    if config['plot']['plots'] is None:
        raise ValueError("The plots key is missing in the yaml file")

    # 讀取畫圖的設定，看有哪些實驗組合是要畫的，綜合起來，我們來 check
    each_plot_combinations, each_plot_pc, all_combinations = read_plot_config(config, args)

    # ============== 讀取 Sacred 的資料夾 check 超參數和 seeds ==============

    if config.get('metrics_json_name', None):
        global METRICS_JSON_NAME
        METRICS_JSON_NAME = config['metrics_json_name']

    comb_to_hp_constraints = check_exp(config, args, list(all_combinations), all_exp_vars)

    # qmix ucb(6,9)_c1_w20000 originalState rs1 curSightInfo
    # comb = ExpVarComb.get_exp_var_comb({'alg': 'qmix', 'map': '8m_vs_9m', 'state': 'originalState', 'rs': 'rs1',
    #                                     'sight': 'ucb(6,9)_c1_w20000',
    #                                     'sight_info': 'curSightInfo'})
    #
    # print('comb_to_hp_constraints:')
    # pprint(comb_to_hp_constraints)
    # print(f'comb_to_hp_constraints.keys():')
    # for key in comb_to_hp_constraints.keys():
    #     print(key)
    # print('comb:',comb)
    # pprint(comb_to_hp_constraints[comb])
    # raise ValueError('stop here')

    if args.verbose:
        pprint(comb_to_hp_constraints)

    # ============== 繪圖 ==============

    if not args.no_plot:
        for plot_save_path in each_plot_combinations.keys():
            pc = each_plot_pc[plot_save_path]
            combinations = each_plot_combinations[plot_save_path]
            plot_one(pc, config, combinations, all_exp_vars, comb_to_hp_constraints, args)


if __name__ == '__main__':
    main()
