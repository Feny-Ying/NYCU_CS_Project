"""
20240805 註：由於現在只整理 sacred 且主要在 name、argv 去改超參，沒有 search 了，所以這個檔案很久沒用，沒維護了。


做完 hyper selection 的實驗後，除了用 ``et_grid_search_best.py`` 來得知什麼組合最好。
我們可能也想知道不同組合的訓練曲線，這時候就可以用這個 script 來畫圖。

請在執行前確認 sacred 有備份。
本檔案會直接操作，為避免出錯，請先備份。

原本的資料夾結構：

method
|_______env
          |_______run1
          |_______run2
          |_______...

執行後的資料夾結構：

method
|_______env
          |_______combination1
          |                 |_______env
          |                           |_______run1
          |                           |_______run2
          |                           |_______...
          |_______combination2
          |                 |_______env
          |                           |_______run1
          |                           |_______run2
          |                           |_______...
          ...

本程式需要的輸入參數：

- ``-f``: method 的資料夾路徑
- ``-env``: 想要畫的環境 (上面架構圖的 env)
- ``--hyper-params``: 分組的 hyperparameters

e.g.,

``python tools/split_dir_by_hyper_comb.py -f projects/hyperparamSelection/sacred\ see\ each\ comb/iql_reset/ -env lbforaging-Foraging-15x15-3p-5f-v1 --hyper-params use_rnn hidden_dim ``


"""
import json
import os
import random


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--method-folder', type=str, required=True,
                        help='The folder containing the results of different hyperparameter combinations.')
    parser.add_argument('-env', '--env', '--environment', type=str, required=True, dest='env',
                        help='The environment to be plotted.')
    parser.add_argument('--hyper-params', type=str, nargs='+', required=True,
                        help='The hyperparameters to be plotted.')
    return parser.parse_args()


if __name__ == '__main__':
    ARGS = get_args()
    method_root_folder = ARGS.method_folder
    hyper_params = ARGS.hyper_params
    env = ARGS.env

    env_root_folder = str(os.path.join(method_root_folder, env))

    # Get all the runs
    for run in os.listdir(env_root_folder):
        if run.startswith('_'):
            continue
        run_folder = os.path.join(env_root_folder, run)
        # Get the hyperparameters
        if not os.path.exists(os.path.join(run_folder, 'config.json')):
            continue
        with open(os.path.join(run_folder, 'config.json'), 'r') as f:
            config = json.load(f)
        # Get the hyperparameters to be compared
        config_to_compare = {k: config[k] for k in hyper_params}
        # Make the folder
        combination_str = ' '.join([f'{k}_{v}' for k, v in config_to_compare.items()])
        combination_folder = os.path.join(env_root_folder, combination_str)
        os.makedirs(combination_folder, exist_ok=True)
        # Get a random str to distinguish the runs
        random_str = str(random.randint(0, 100000000))
        new_run_folder = os.path.join(combination_folder, env, random_str)
        assert not os.path.exists(
            new_run_folder), f'{new_run_folder} already exists; the random name is conflict; you may rerun the script.'
        os.makedirs(os.path.join(combination_folder, env), exist_ok=True)
        # Move the run
        os.rename(run_folder, os.path.join(combination_folder, env, random_str))
