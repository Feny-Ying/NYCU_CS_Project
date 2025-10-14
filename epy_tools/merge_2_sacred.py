"""例如：在NV練的sacred載下來後和本地的sacred合併，但因為 run name 會打架，
所以需要將新的資料夾的run name修改成比舊的最大數字大。

假設sacred資料夾結構如下： (method = alg_sight)
    - 第 1 層：method1, method2, method3, ...
    - 第 2 層：env1, env2, env3, ...
    - 第 3 層：run1, run2, run3, ... (run的編號不一定是seed，順序可能是亂的); 這層略過名字前面有 `_` 的資料夾

"""

import argparse
import os
import shutil

PARSER = argparse.ArgumentParser()
PARSER.add_argument('-s', '-source', '--source_dir', type=str, help='其它的sacred資料夾路徑, e.g., ~/Download/sacred/')
PARSER.add_argument('-d', '-dest', '--output_dir', type=str, help='主要的sacred資料夾路徑, e.g., POEM/sacred/')

ARGS = PARSER.parse_args()

# 獲取所有子資料夾
methods = [d for d in os.listdir(ARGS.source_dir) if os.path.isdir(os.path.join(ARGS.source_dir, d))]

for method in methods:
    # Check if the method is existed in the destination
    # If not, just copy the whole folder
    if not os.path.exists(os.path.join(ARGS.output_dir, method)):
        shutil.copytree(os.path.join(ARGS.source_dir, method), os.path.join(ARGS.output_dir, method))
        print(f'Copy {os.path.join(ARGS.source_dir, method)} to {os.path.join(ARGS.output_dir, method)}')
        continue

    # If the method is existed in the destination, then check the envs
    envs = [d for d in os.listdir(os.path.join(ARGS.source_dir, method)) if
            os.path.isdir(os.path.join(ARGS.source_dir, method, d))]
    for env in envs:
        # Check if the env is existed in the destination
        # If not, just copy the whole folder
        if not os.path.exists(os.path.join(ARGS.output_dir, method, env)):
            shutil.copytree(os.path.join(ARGS.source_dir, method, env), os.path.join(ARGS.output_dir, method, env))
            print(f'Copy {os.path.join(ARGS.source_dir, method, env)} to {os.path.join(ARGS.output_dir, method, env)}'
                  )
            continue
        # Env dir is existed in the destination
        # Check the runs, get the max number of the runs
        runs = [d for d in os.listdir(os.path.join(ARGS.source_dir, method, env)) if
                os.path.isdir(os.path.join(ARGS.source_dir, method, env, d))]
        # Exclude the dir name starts with `_`
        runs = [d for d in runs if not d.startswith('_')]
        if len(runs) == 0:
            raise ValueError(f'No runs in {os.path.join(ARGS.source_dir, method, env)}, so strange!')
        # Get the max number of the runs in the destination
        max_run = max([int(run) for run in os.listdir(os.path.join(ARGS.output_dir, method, env)) if
                       os.path.isdir(os.path.join(ARGS.output_dir, method, env, run)) and not run.startswith('_')])
        # Copy the runs to the destination
        for run in runs:
            shutil.copytree(os.path.join(ARGS.source_dir, method, env, run),
                            os.path.join(ARGS.output_dir, method, env, str(max_run + 1)),
                            dirs_exist_ok=True)
            print(
                f'Copy {os.path.join(ARGS.source_dir, method, env, run)} to {os.path.join(ARGS.output_dir, method, env, str(max_run + 1))}')

            max_run += 1
