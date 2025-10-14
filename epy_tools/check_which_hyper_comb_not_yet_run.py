"""
這個程式是用來檢查哪些超參數組合還沒有被執行過。
舉例來說，如果我今天在兩台機器跑 72 個組合，結果兩邊加起來只有 68 個資料夾，那就代表有 4 個組合還沒有被執行過。
用這個程式就可以找出哪些組合還沒有被執行過。

輸入：
- Sacred 資料夾路徑
- 各項超參數 (dict of list)
    - 請用 "a:[value1,value2, ...]" "b:[value1,value2, ...]" 的格式

這個程式會檢查每個資料夾的 config.json 來看缺了啥。
"""

import argparse
import json
import os
from itertools import product


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', type=str, required=True, help='Path to the Sacred folder')
    parser.add_argument('-p', '--params', type=str, required=True, nargs='+',
                        help='Hyperparameters. E.g., "-p \"use_rnn:[True,False]\"')
    return parser.parse_args()


if __name__ == '__main__':
    ARGS = get_args()

    file_dir = ARGS.folder
    params = ARGS.params

    params_dict = {}
    for param in params:
        assert ':' in param, f'Invalid format: {param}'
        key, value = param.split(':')
        assert value.startswith('[') and value.endswith(']')
        value = value.split('[')[1].split(']')[0].split(',')
        params_dict[key] = value

    # Get all the results
    results = []
    for exp_dir in os.listdir(file_dir):
        exp_dir_path = os.path.join(file_dir, exp_dir)
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

        this_exp_comb = {}
        for param in params_dict:
            # Check if the param is in config_data
            assert param in config_data, f'{param} not in {config_data}'
            this_value = config_data[param]

            # Check if the value is in params_dict[param]
            # 把 params_dict[param] 裡面的值的型態轉換成跟 this_value 一樣
            # 假設型別不會亂變，我們直接看 config.json 的型別來更新 params_dict[param] 裡面的值的型別
            found = False
            for value in params_dict[param]:
                # 讓 value 的型態跟 this_value 一樣
                if isinstance(this_value, bool):
                    if isinstance(value, str):
                        value = value.lower() == 'true'
                elif isinstance(this_value, int):
                    value = int(value)
                elif isinstance(this_value, float):
                    value = float(value)
                elif isinstance(this_value, str):
                    value = str(value)
                else:
                    assert False, f'Unknown type: {type(this_value)}'
                if this_value == value:
                    found = True
                    break
            assert found, f'{this_value} not in {params_dict[param]}'


            # 更新 params_dict[param] 裡面的型態

            def get_type_map_(type_):
                def change_type(value_):
                    if isinstance(value_, str):
                        if value_.lower() == 'true':
                            value_ = True
                        elif value_.lower() == 'false':
                            value_ = False

                    return type_(value_)

                return change_type


            params_dict[param] = list(map(get_type_map_(type(this_value)), params_dict[param]))
            #
            this_exp_comb[param] = config_data[param]

        results.append(this_exp_comb)

    print(f'Hyperparameters: {params_dict}')

    # Check if all combinations are in results
    # E.g., 假設有兩個超參數 {'weight_decay': ['1e-5', '1e-3', '1e-1', '1e1'], 'hidden_dim': ['64', '128', '256']}
    #       則應該要有 4*3=12 種組合 (weight_decay=1e-5, hidden_dim=64), (weight_decay=1e-5, hidden_dim=128), ...
    #       這個迴圈就是檢查有沒有缺少哪些組合
    #
    # 先建一個有所有組合的 tuple (順序照 params_dict.keys())
    # 再迴圈每個組合，如果有在 results 裡面就把它從 all_combinations 裡面移除
    # 最後 all_combinations 裡面剩下的就是還沒有被執行過的組合
    all_combinations = list(product(*params_dict.values()))
    print(f'Number of all combinations: {len(all_combinations)}, but got {len(results)} results')

    # Check if all combinations are in results
    for exp_comb in results:
        this_comb = tuple([exp_comb[key] for key in params_dict.keys()])
        if this_comb in all_combinations:
            all_combinations.remove(this_comb)

    print(f'Combinations not yet run:')
    print('=========================')
    print('\t'.join(params_dict.keys()))
    for comb in all_combinations:
        for i, key in enumerate(params_dict.keys()):
            print(f'{comb[i]}', end='\t')
        print()
