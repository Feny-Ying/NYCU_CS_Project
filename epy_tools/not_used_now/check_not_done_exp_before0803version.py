"""這是個用來檢查所有`方法`x`環境`x`seeds`的實驗是否都已經完成的工具。


Input:
    - Path to the experiment folder (sacred)
    - Environment names
    - Seeds: list of integers

至於方法名稱，系統會自動從你給的 sacred 資料夾下一層的資料夾名稱中抓取。



假設資料夾結構如下：`sacred/` → `${alg+sight}` → `${env}/` → `${runs}/`
    - 第 1 層：method1, method2, method3, ... (這裡的 method 指的是 alg+sight)
    - 第 2 層：env1, env2, env3, ...
    - 第 3 層：run1, run2, run3, ... (run的編號不一定是seed，順序可能是亂的); 這層略過名字前面有 `_` 的資料夾

最後，畫一個 summary table，告訴你哪些方法的哪些環境的哪些seed還沒有完成。
Row 為方法名稱，Column 為環境名稱，Cell 為還沒有完成的 seed。
"""
import argparse
import json
import os
from pprint import pprint

import pandas as pd

def get_method_name(alg_name, sight_name):
    return f'{alg_name}_{sight_name}'


def get_args():
    parser = argparse.ArgumentParser()
    # Required
    parser.add_argument('-f', '--folder', type=str, required=True, help='Path to the Sacred folder')
    parser.add_argument('-envs', '--envs', type=str, required=True, nargs='+', help='Environment names')
    parser.add_argument('-seed', '--seeds', type=int, required=True, nargs='+', help='Seeds')
    # Optional
    parser.add_argument('-v', '--verbose', action='store_true', help='Print verbose information')
    parser.add_argument('-ex', '--exclude', type=str, nargs='+', default=[],
                        help='Exclude some method names containing these words')
    return parser.parse_args()


def main():
    args = get_args()

    # Get method names
    method_names = [d for d in os.listdir(args.folder) if os.path.isdir(os.path.join(args.folder, d))]

    # Exclude some method names
    if len(args.exclude) > 0:
        method_names = [method_name for method_name in method_names if
                        not any(exclude_word in method_name for exclude_word in args.exclude)]

    # Sort the method names alphabetically
    method_names = sorted(method_names)

    if args.verbose:
        print(f'Founded method names:')
        pprint(method_names)

    # Create an empty DataFrame with method names as row index and environment names as column index
    summary_table_not_done_seeds = pd.DataFrame(index=method_names, columns=args.envs)

    #
    done_methods = []

    # Check each method's envs and seeds
    for method_name in method_names:
        method_dir = os.path.join(args.folder, method_name)
        env_names = [d for d in os.listdir(method_dir) if os.path.isdir(os.path.join(method_dir, d))]

        # Only check the envs that you want
        env_names = [env_name for env_name in env_names if env_name in args.envs]

        if args.verbose:
            print(f'Founded env names for {method_name}:')
            pprint(env_names)

        this_method_done = True
        for env_name in env_names:
            env_dir = os.path.join(method_dir, env_name)
            run_names = [d for d in os.listdir(env_dir) if os.path.isdir(os.path.join(env_dir, d))]
            run_names = [run_name for run_name in run_names if not run_name.startswith('_')]

            # Sort the run names by str length and str alphabetically
            run_names = sorted(run_names, key=lambda x: (len(x), x))

            if args.verbose:
                print(f'Founded run names for {method_name}/{env_name}:')
                pprint(run_names)

            # Get each run's config.json to check if the seed is done
            found_seeds = []
            for run_name in run_names:
                run_dir = os.path.join(env_dir, run_name)
                config_path = os.path.join(run_dir, 'config.json')
                if not os.path.exists(config_path):
                    raise FileNotFoundError(f'config.json not found in {run_dir}')
                config_data = json.load(open(config_path, 'r'))
                if args.verbose:
                    print(f'Checking config_data for {method_name}/{env_name}/{run_name}:')

                # Check if the seed is done
                seed = config_data['seed']

                # Check whether there are duplicated seeds
                if seed in found_seeds:
                    raise ValueError(f'Duplicated seed: {seed} in {method_name} / {env_name} (run_name: {run_name})')

                # Check whether the seed is NOT in your wish seeds
                if seed not in args.seeds:
                    raise ValueError(f'Unexpected seed: {seed} in {method_name}/{env_name} (run_name: {run_name})')

                found_seeds.append(seed)

            not_done_seeds = list(set(args.seeds) - set(found_seeds))
            summary_table_not_done_seeds.loc[method_name, env_name] = not_done_seeds

            if len(not_done_seeds) > 0:
                this_method_done = False

        # If the row has NaN (not found) or non-empty list (some seeds are not done), then this method is not done
        if summary_table_not_done_seeds.loc[method_name].isnull().any():
            this_method_done = False
        elif summary_table_not_done_seeds.loc[method_name].apply(lambda x: len(x) > 0).any():
            this_method_done = False

        if this_method_done:
            done_methods.append(method_name)

    # Set df to print all the content
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    #
    print(f'Done methods:')
    print(done_methods)

    # Remove rows which only has empty lists
    summary_table_not_done_seeds = summary_table_not_done_seeds[
        summary_table_not_done_seeds.apply(lambda x: any(x), axis=1)]

    print('Summary table for not done seeds:')
    print(summary_table_not_done_seeds)

    # Show successful message if all methods are done
    if summary_table_not_done_seeds.empty:
        print('>>>>>> All methods are done! <<<<<<')


if __name__ == '__main__':
    main()
