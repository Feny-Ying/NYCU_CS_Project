"""
簡單講，就是把 iql/xxx1s xxx2s ... -> iql_1s/xxx1s iql_2s/xxx2s ...


適用於練完 pure sight 且沒在 name 加註 ?s。此時可用這個程式將資料夾分成不同視野範圍。
例如: iql/ 的 ....2s.... -> iql_2s/...2s



ChatGPT conversation:



寫個python
```
❯ tree vdn -L 1
vdn
├── lbforaging-Foraging-15s-15x15-3p-5f-v1
├── lbforaging-Foraging-15s-15x15-4p-5f-coop-v1
├── lbforaging-Foraging-15s-15x15-4p-5f-v1
├── lbforaging-Foraging-2s-15x15-3p-5f-v1
├── lbforaging-Foraging-2s-15x15-4p-5f-coop-v1
├── lbforaging-Foraging-2s-15x15-4p-5f-v1
├── lbforaging-Foraging-3s-15x15-3p-5f-v1
├── lbforaging-Foraging-3s-15x15-4p-5f-coop-v1
├── lbforaging-Foraging-3s-15x15-4p-5f-v1
├── lbforaging-Foraging-4s-15x15-3p-5f-v1
├── lbforaging-Foraging-4s-15x15-4p-5f-coop-v1
├── lbforaging-Foraging-4s-15x15-4p-5f-v1
├── lbforaging-Foraging-5s-15x15-3p-5f-v1
├── lbforaging-Foraging-5s-15x15-4p-5f-coop-v1
├── lbforaging-Foraging-5s-15x15-4p-5f-v1
├── lbforaging-Foraging-6s-15x15-3p-5f-v1
├── lbforaging-Foraging-6s-15x15-4p-5f-coop-v1
└── lbforaging-Foraging-6s-15x15-4p-5f-v1

18 directories, 0 files

```
把資料夾內依以下規則複製到適當處

`vdn/lbforaging-Foraging-2s-15x15-3p-5f-v1` => `vdn_2s/lbforaging-Foraging-2s-15x15-3p-5f-v1`

基本上就是： `vdn/...{??s}...` => `vdn_{??s}/.../

For LBF

"""

import argparse
import os
import shutil

PARSER = argparse.ArgumentParser()
PARSER.add_argument('-d', '--source_dir', type=str, help='原始資料夾路徑, e.g., sacred/qmix/')
PARSER.add_argument('-od', '--output_dir', type=str, help='輸出root資料夾路徑, e.g., POEM/sacred/')

ARGS = PARSER.parse_args()

# 獲取所有子資料夾
subdirs = [d for d in os.listdir(ARGS.source_dir) if os.path.isdir(os.path.join(ARGS.source_dir, d))]

root_dir = ARGS.source_dir.strip('/').split('/')[-1]  # e.g., 'vdn', 'iql', 'qmix'

for subdir in subdirs:
    print(f'subdir: {subdir}')
    if 'lbforaging' not in subdir:  # subdir e.g., "lbforaging-Foraging-15s-15x15-4p-5f-v1"
        print(f"跳過 {subdir}")
        continue
    # 尋找 `??s` 格式的子資料夾
    if 's-' in subdir:
        # 擷取 `??s` 部分

        if 's' in subdir.split('-')[2]:
            sight_str = subdir.split('-')[2]
        elif 's' in subdir.split('-')[1]:
            sight_str = subdir.split('-')[1]
        else:
            raise ValueError(f"找不到視野範圍: {subdir}")


        # 設定目標資料夾
        target_dir = f'{root_dir}_{sight_str}/'

        # New dir name (置換 ??S 成 args.new_sight)
        new_dir = subdir

        # 20240805: This may be bugged. It creates non-used dirs.
        # # 確保目標資料夾存在
        # if not os.path.exists(target_dir):
        #     os.makedirs(target_dir)

        # # 複製資料夾到目標位置
        # shutil.copytree(os.path.join(ARGS.source_dir, subdir), os.path.join(ARGS.output_dir, target_dir, new_dir))

        source_env_dir = subdir
        target_env_dir = new_dir
        method_dir = target_dir

        target_runs_dir = os.path.join(ARGS.output_dir, method_dir, target_env_dir)

        source_runs_dir = os.path.join(ARGS.source_dir, source_env_dir)

        # 看目標是否存在，已別的實驗結果在裡面。有的話照裡面最大的數字加下去
        # from 0
        if os.path.exists(target_runs_dir):
            # 找出原本存在的最大的數字
            min_ok_trial = 1
            for d in os.listdir(target_runs_dir):
                if d == '_sources':
                    continue
                trial = int(d)
                assert trial >= 1, f'Assume the trial number starts from 1, but got: {trial}'
                if trial >= min_ok_trial:
                    min_ok_trial = trial + 1
        else:
            min_ok_trial = 1
        # 列出 source runs，並按照數字排序
        source_runs = [d for d in os.listdir(source_runs_dir) if d != '_sources']
        source_runs = sorted([int(d) for d in source_runs])
        print(f"source_runs: {source_runs}")
        # 複製資料夾到目標位置，並且改名
        for source_run_num in source_runs:
            source_path = os.path.join(source_runs_dir, str(source_run_num))
            target_path = os.path.join(target_runs_dir, str(min_ok_trial))
            shutil.copytree(source_path, target_path)
            min_ok_trial += 1

        print("資料夾複製完成。")
