"""
epy 練完會有 sacred, tb_logs, models 三個母資料夾。
由於它把不同實驗資料夾混在一起，我們想要把它們分開。

20240805 註：由於現在只整理 sacred，所以這個檔案很久沒維護了。



```
.
├── models
│   ├── iql_reset_seed0_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-09 21-54-20.497599
│   ├── iql_reset_seed0_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 04-12-59.290058
│   ├── iql_reset_seed0_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 12-48-04.845584
│   ├── iql_reset_seed0_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 12-49-10.152123
│   ├── iql_reset_seed0_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 16-39-35.659632
│   ├── iql_reset_seed0_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 17-10-44.725803
│   ├── iql_reset_seed1_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 04-08-18.931334
│   ├── iql_reset_seed1_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 04-08-55.858299
│   ├── iql_reset_seed1_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 12-48-04.916869
│   ├── iql_reset_seed1_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 12-49-10.308472
│   ├── iql_reset_seed1_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 16-25-16.718470
│   ├── iql_reset_seed1_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 17-00-38.862833
│   ├── iql_reset_seed2_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-09 21-54-16.411859
│   ├── iql_reset_seed2_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 04-12-39.382723
│   ├── iql_reset_seed2_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 12-48-04.709515
│   ├── iql_reset_seed2_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 12-48-04.833415
│   ├── iql_reset_seed2_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 12-49-10.102282
│   ├── iql_reset_seed2_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 12-49-10.175478
│   ├── iql_reset_seed3_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-09 21-54-16.170164
│   ├── iql_reset_seed3_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-09 21-54-20.257678
│   ├── iql_reset_seed3_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 12-48-04.825961
│   ├── iql_reset_seed3_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 12-49-10.105762
│   ├── iql_reset_seed3_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 16-40-03.561933
│   ├── iql_reset_seed3_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 17-10-49.724660
│   ├── iql_reset_seed4_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-09 21-54-16.428025
│   ├── iql_reset_seed4_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 04-09-25.416582
│   ├── iql_reset_seed4_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 16-25-16.714796
│   ├── iql_reset_seed4_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 16-41-46.974641
│   ├── iql_reset_seed4_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 17-02-34.949932
│   └── iql_reset_seed4_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 17-11-32.108244
├── sacred
│   └── iql_reset
│       └── lbforaging-Foraging-15x15-4p-3f-v1
│           ├── 1
│           ├── 10
│           ├── 11
│           ├── 12
│           ├── 13
│           ├── 14
│           ├── 15
│           ├── 16
│           ├── 17
│           ├── 18
│           ├── 19
│           ├── 2
│           ├── 20
│           ├── 21
│           ├── 22
│           ├── 23
│           ├── 24
│           ├── 25
│           ├── 26
│           ├── 27
│           ├── 28
│           ├── 29
│           ├── 3
│           ├── 30
│           ├── 4
│           ├── 5
│           ├── 6
│           ├── 7
│           ├── 8
│           ├── 9
│           └── _sources
└── tb_logs
    ├── iql_reset_seed0_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-09 21-54-20.497599
    ├── iql_reset_seed0_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 04-12-59.290058
    ├── iql_reset_seed0_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 12-48-04.845584
    ├── iql_reset_seed0_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 12-49-10.152123
    ├── iql_reset_seed0_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 16-39-35.659632
    ├── iql_reset_seed0_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 17-10-44.725803
    ├── iql_reset_seed1_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 04-08-18.931334
    ├── iql_reset_seed1_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 04-08-55.858299
    ├── iql_reset_seed1_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 12-48-04.916869
    ├── iql_reset_seed1_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 12-49-10.308472
    ├── iql_reset_seed1_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 16-25-16.718470
    ├── iql_reset_seed1_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 17-00-38.862833
    ├── iql_reset_seed2_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-09 21-54-16.411859
    ├── iql_reset_seed2_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 04-12-39.382723
    ├── iql_reset_seed2_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 12-48-04.709515
    ├── iql_reset_seed2_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 12-48-04.833415
    ├── iql_reset_seed2_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 12-49-10.102282
    ├── iql_reset_seed2_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 12-49-10.175478
    ├── iql_reset_seed3_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-09 21-54-16.170164
    ├── iql_reset_seed3_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-09 21-54-20.257678
    ├── iql_reset_seed3_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 12-48-04.825961
    ├── iql_reset_seed3_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 12-49-10.105762
    ├── iql_reset_seed3_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 16-40-03.561933
    ├── iql_reset_seed3_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 17-10-49.724660
    ├── iql_reset_seed4_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-09 21-54-16.428025
    ├── iql_reset_seed4_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 04-09-25.416582
    ├── iql_reset_seed4_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 16-25-16.714796
    ├── iql_reset_seed4_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 16-41-46.974641
    ├── iql_reset_seed4_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 17-02-34.949932
    └── iql_reset_seed4_lbforaging-Foraging-15x15-4p-3f-v1_2024-04-10 17-11-32.108244

```

 iql_reset 指的是 method
lbforaging-Foraging-15x15-4p-3f-v1 是 env
sacred / method / env / 下面的那些資料夾(開頭不含"_")的是不同實驗的結果。每個資料夾對應到 models 下和 tb_logs 下的資料夾(數字順序對應到那邊的檔案名裡的日期順序)
每個資料夾含有 config.json。

寫一個python script，
輸入：root_dir, hyperparameters (>=1個) 用argparse吃
效果：
              依照 hyper combination 把原本 root dir 的 檔案們分開

root_dir
|_ ...

split_root_dir
|_ hyperComb1
|                         |_ sacred
|                         |_ models
|                         |_ tb_logs
|_ hyperComb1
|                         |_ sacred
|                         |_ models
|                         |_ tb_logs
| ...



"""
import argparse
import json
import os
import shutil


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--root-dir', type=str, required=True, help='The root directory')
    parser.add_argument('-hyper', '--hyper-params', type=str, nargs='+', required=True, help='Hyperparameters to split')
    parser.add_argument('-o', '--output-dir', type=str, required=True, help='The output directory')
    parser.add_argument('-s', '--insert-str', type=str, required=False, default='',
                        help='The string to be inserted between the method and the hyperparameter combination.')
    parser.add_argument('--sacred-only', action='store_true', help='Only split sacred folders')
    parser.add_argument('--skip-models', action='store_true', help='Skip models folders')
    return parser.parse_args()


if __name__ == '__main__':
    ARGS = get_args()
    ROOT_DIR = ARGS.root_dir

    # Output dir should be created if not exists
    if not os.path.exists(ARGS.output_dir):
        os.makedirs(ARGS.output_dir)

    # Step: 對應 sacre, models, tb_logs 裡的資料夾。同一實驗的對在一起。
    # `sacred` 下一層是 method，再下一層是 env，再下一層是實驗資料夾(開頭不含"_")(稱為 run)
    # `tb_logs` 下一層直接是實驗資料夾，資料夾名含有 method, seed, env
    # `models` 下一層直接是實驗資料夾，資料夾名含有 method, seed, env

    # 每個 tb_logs 和 models 下的資料夾檔名含有時間，把時間的部分拿來排序，排序後的順序對應到 `sacred` 下的 run 的名稱數字順序
    # 這樣就可以把同一實驗的東西對應在一起了。

    # 先知道共有哪些 method, env 要處理的

    method_env_pairs = []
    for method in os.listdir(os.path.join(ROOT_DIR, 'sacred')):
        for env in os.listdir(os.path.join(ROOT_DIR, 'sacred', method)):
            method_env_pairs.append((method, env))

    # 依照 method, env 名稱，把 tb_logs 和 models 下的資料夾檔名含有此 method, env 的資料夾拿出來
    for method, env in method_env_pairs:

        # 換來看 sacred
        sacred_dir = os.path.join(ROOT_DIR, 'sacred', method, env)
        sacred_folders = os.listdir(sacred_dir)
        # 拿掉開頭是 "_" 的資料夾
        sacred_folders = [folder for folder in sacred_folders if not folder.startswith('_')]
        # 先用字串長度，再用字母排序
        sorted_sacred_folders = sorted(sacred_folders, key=lambda x: (len(x), x))

        if not ARGS.sacred_only:
            # 看 tb_logs 下的資料夾
            tb_logs_dir = os.path.join(ROOT_DIR, 'tb_logs')
            # 篩掉不含 method, env 的資料夾
            tb_logs_folders = [folder for folder in os.listdir(tb_logs_dir) if method in folder and env in folder]
            # 用時間排序 tb_logs 下的資料夾
            sorted_tb_logs_folders = sorted(tb_logs_folders, key=lambda x: x.split('_')[-1])

            if not ARGS.skip_models:
                # 看 models 下的資料夾
                models_dir = os.path.join(ROOT_DIR, 'models')
                # 篩掉不含 method, env 的資料夾
                models_folders = [folder for folder in os.listdir(models_dir) if method in folder and env in folder]
                # 用時間排序 models 下的資料夾
                sorted_models_folders = sorted(models_folders, key=lambda x: x.split('_')[-1])
            else:
                sorted_models_folders = [None] * len(sorted_sacred_folders)
        else:
            # Make None for tb_logs and models
            sorted_tb_logs_folders = [None] * len(sorted_sacred_folders)
            sorted_models_folders = [None] * len(sorted_sacred_folders)

        # 來讀 config.json
        for sacred_dir, tb_log_dir, models_dir in zip(sorted_sacred_folders, sorted_tb_logs_folders,
                                                      sorted_models_folders):
            print(f'Processing {method} {env} {sacred_dir}')
            # 這裡可以讀 config.json 了
            with open(os.path.join(ROOT_DIR, 'sacred', method, env, sacred_dir, 'config.json'), 'r') as f:
                config = json.load(f)
            # 這裡可以把 config.json 的 hyperparameters 拿來分組了
            hyper_comb = '_'.join(tuple(str(config[hyper]) for hyper in ARGS.hyper_params))

            # 這裡可以把資料夾"複製"到新的地方了
            new_root_dir = ARGS.output_dir
            if len(ARGS.insert_str) > 0:
                new_method = f'{method}_{ARGS.insert_str}_{hyper_comb}'
            else:
                new_method = f'{method}_{hyper_comb}'

            os.makedirs(new_root_dir, exist_ok=True)
            # 在下面建立 sacred, tb_logs, models 的資料夾，如同原本的結構，只是沒有別的hyper_comb的資料夾來干擾
            os.makedirs(os.path.join(new_root_dir, 'sacred', new_method, env), exist_ok=True)
            if not ARGS.sacred_only:
                os.makedirs(os.path.join(new_root_dir, 'tb_logs', new_method, env, ), exist_ok=True)
                if not ARGS.skip_models:
                    os.makedirs(os.path.join(new_root_dir, 'models', new_method, env), exist_ok=True)
            # 把原本的資料夾複製過去，用 shutil.copytree
            shutil.copytree(str(os.path.join(ROOT_DIR, 'sacred', method, env, sacred_dir)),
                            str(os.path.join(new_root_dir, 'sacred', new_method, env, sacred_dir)))
            # 雖然原本的 models, tb_logs 的資料夾名稱沒有 method, env，新的要有
            if not ARGS.sacred_only:
                os.makedirs(os.path.join(new_root_dir, 'tb_logs', new_method, env), exist_ok=True)
                if not ARGS.skip_models:
                    os.makedirs(os.path.join(new_root_dir, 'models', new_method, env), exist_ok=True)
                # 把 models, tb_logs 的資料夾複製過去
                shutil.copytree(str(os.path.join(ROOT_DIR, 'tb_logs', tb_log_dir)),
                                str(os.path.join(new_root_dir, 'tb_logs', new_method, env, tb_log_dir)))
                if not ARGS.skip_models:
                    shutil.copytree(str(os.path.join(ROOT_DIR, 'models', models_dir)),
                                    str(os.path.join(new_root_dir, 'models', new_method, env, models_dir)))

            # # 在下面建立 sacred, tb_logs, models 的資料夾，如同原本的結構，只是沒有別的hyper_comb的資料夾來干擾
            # os.makedirs(os.path.join(new_root_dir, 'sacred'), exist_ok=True)
            # os.makedirs(os.path.join(new_root_dir, 'tb_logs'), exist_ok=True)
            # os.makedirs(os.path.join(new_root_dir, 'models'), exist_ok=True)
            # # 新的 sacred 還是要照原本的結構，要有 method, env
            # os.makedirs(os.path.join(new_root_dir, 'sacred', method, env), exist_ok=True)
            # # 把原本的資料夾複製過去，用 shutil.copytree
            # shutil.copytree(str(os.path.join(ROOT_DIR, 'sacred', method, env, sacred_dir)),
            #                 str(os.path.join(new_root_dir, 'sacred', method, env, sacred_dir)))
            # # 雖然原本的 models, tb_logs 的資料夾名稱沒有 method, env，新的要有
            # os.makedirs(os.path.join(new_root_dir, 'tb_logs', method, env), exist_ok=True)
            # os.makedirs(os.path.join(new_root_dir, 'models', method, env), exist_ok=True)
            # # 把 models, tb_logs 的資料夾複製過去
            # shutil.copytree(str(os.path.join(ROOT_DIR, 'tb_logs', tb_log_dir)),
            #                 str(os.path.join(new_root_dir, 'tb_logs', method, env, tb_log_dir)))
            # shutil.copytree(str(os.path.join(ROOT_DIR, 'models', models_dir)),
            #                 str(os.path.join(new_root_dir, 'models', method, env, models_dir)))
