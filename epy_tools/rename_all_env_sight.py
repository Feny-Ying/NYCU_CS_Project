"""
❯ tree -d -L 2
.
├── iql_1s
│   ├── resco_benchmark-cologne3-3s-v1
│   ├── resco_benchmark-cologne3fs10-3s-v1
│   └── resco_benchmark-cologne3fs5-3s-v1
├── iql_1s_hidden256
│   ├── resco_benchmark-cologne3-1s-v1
│   ├── resco_benchmark-cologne3fs10-1s-v1
│   └── resco_benchmark-cologne3fs5-1s-v1
├── iql_2s
│   ├── resco_benchmark-cologne3-3s-v1
│   ├── resco_benchmark-cologne3fs10-2s-v1
│   └── resco_benchmark-cologne3fs5-2s-v1
├── iql_2s_hidden256
│   ├── resco_benchmark-cologne3-2s-v1
│   ├── resco_benchmark-cologne3fs10-2s-v1
│   └── resco_benchmark-cologne3fs5-2s-v1
├── iql_3s
│   ├── resco_benchmark-cologne3-3s-v1
│   ├── resco_benchmark-cologne3fs10-3s-v1
│   └── resco_benchmark-cologne3fs5-3s-v1
├── iql_3s_hidden256
│   ├── resco_benchmark-cologne3-3s-v1
│   ├── resco_benchmark-cologne3fs10-3s-v1
│   └── resco_benchmark-cologne3fs5-3s-v1
├── iql_equal(1,2,3)~5e6
│   ├── resco_benchmark-cologne3-3s-v1
│   ├── resco_benchmark-cologne3fs10-3s-v1
│   └── resco_benchmark-cologne3fs5-3s-v1
├── iql_equal(1,2,3)~5e6_gradNext
│   ├── resco_benchmark-cologne3-3s-v1
│   ├── resco_benchmark-cologne3fs10-3s-v1
│   └── resco_benchmark-cologne3fs5-3s-v1
├── iql_equal(1,2,3)~5e6_gradNext_hidden256
│   ├── resco_benchmark-cologne3-3s-v1
│   ├── resco_benchmark-cologne3fs10-3s-v1
│   └── resco_benchmark-cologne3fs5-3s-v1
├── iql_equal(1,2,3)~5e6_hidden256
│   ├── resco_benchmark-cologne3-3s-v1
│   ├── resco_benchmark-cologne3fs10-3s-v1
│   └── resco_benchmark-cologne3fs5-3s-v1
├── qmix_1s
│   ├── resco_benchmark-cologne3-1s-v1
│   ├── resco_benchmark-cologne3fs10-1s-v1
│   └── resco_benchmark-cologne3fs5-1s-v1
├── qmix_1s_hidden256
│   ├── resco_benchmark-cologne3-1s-v1
│   ├── resco_benchmark-cologne3fs10-1s-v1
│   └── resco_benchmark-cologne3fs5-1s-v1
├── qmix_2s
│   ├── resco_benchmark-cologne3-2s-v1
│   ├── resco_benchmark-cologne3fs10-2s-v1
│   └── resco_benchmark-cologne3fs5-2s-v1
├── qmix_2s_hidden256
│   ├── resco_benchmark-cologne3-2s-v1
│   ├── resco_benchmark-cologne3fs10-2s-v1
│   └── resco_benchmark-cologne3fs5-2s-v1
├── qmix_3s
│   ├── resco_benchmark-cologne3-3s-v1
│   ├── resco_benchmark-cologne3fs10-3s-v1
│   └── resco_benchmark-cologne3fs5-3s-v1
├── qmix_3s_hidden256
│   ├── resco_benchmark-cologne3-3s-v1
│   ├── resco_benchmark-cologne3fs10-3s-v1
│   └── resco_benchmark-cologne3fs5-3s-v1
├── qmix_equal(1,2,3)~5e6
│   ├── resco_benchmark-cologne3-3s-v1
│   ├── resco_benchmark-cologne3fs10-3s-v1
│   └── resco_benchmark-cologne3fs5-3s-v1
├── qmix_equal(1,2,3)~5e6_gradNext
│   ├── resco_benchmark-cologne3-3s-v1
│   ├── resco_benchmark-cologne3fs10-3s-v1
│   └── resco_benchmark-cologne3fs5-3s-v1
├── qmix_equal(1,2,3)~5e6_gradNext_hidden256
│   ├── resco_benchmark-cologne3-3s-v1
│   ├── resco_benchmark-cologne3fs10-3s-v1
│   └── resco_benchmark-cologne3fs5-3s-v1
├── qmix_equal(1,2,3)~5e6_hidden256
│   ├── resco_benchmark-cologne3-3s-v1
│   ├── resco_benchmark-cologne3fs10-3s-v1
│   └── resco_benchmark-cologne3fs5-3s-v1
├── vdn_1s
│   ├── resco_benchmark-cologne3-1s-v1
│   ├── resco_benchmark-cologne3fs10-1s-v1
│   └── resco_benchmark-cologne3fs5-1s-v1
├── vdn_1s_hidden256
│   ├── resco_benchmark-cologne3-1s-v1
│   ├── resco_benchmark-cologne3fs10-1s-v1
│   └── resco_benchmark-cologne3fs5-1s-v1
├── vdn_2s
│   ├── resco_benchmark-cologne3-2s-v1
│   ├── resco_benchmark-cologne3fs10-2s-v1
│   └── resco_benchmark-cologne3fs5-2s-v1
├── vdn_2s_hidden256
│   ├── resco_benchmark-cologne3-2s-v1
│   ├── resco_benchmark-cologne3fs10-2s-v1
│   └── resco_benchmark-cologne3fs5-2s-v1
├── vdn_3s
│   ├── resco_benchmark-cologne3-3s-v1
│   ├── resco_benchmark-cologne3fs10-3s-v1
│   └── resco_benchmark-cologne3fs5-3s-v1
├── vdn_3s_hidden256
│   ├── resco_benchmark-cologne3-3s-v1
│   ├── resco_benchmark-cologne3fs10-3s-v1
│   └── resco_benchmark-cologne3fs5-3s-v1
├── vdn_equal(1,2,3)~5e6
│   ├── resco_benchmark-cologne3-3s-v1
│   ├── resco_benchmark-cologne3fs10-3s-v1
│   └── resco_benchmark-cologne3fs5-3s-v1
├── vdn_equal(1,2,3)~5e6_gradNext
│   ├── resco_benchmark-cologne3-3s-v1
│   ├── resco_benchmark-cologne3fs10-3s-v1
│   └── resco_benchmark-cologne3fs5-3s-v1
├── vdn_equal(1,2,3)~5e6_gradNext_hidden256
│   ├── resco_benchmark-cologne3-3s-v1
│   ├── resco_benchmark-cologne3fs10-3s-v1
│   └── resco_benchmark-cologne3fs5-3s-v1
└── vdn_equal(1,2,3)~5e6_hidden256
    ├── resco_benchmark-cologne3-3s-v1
    ├── resco_benchmark-cologne3fs10-3s-v1
    └── resco_benchmark-cologne3fs5-3s-v1

120 directories


iql, vdn, ... 那層是alg
resco_bencmark-...-v1 那層是 env
如何寫一個python，把env的名字是2s 1s 的全改成3s
(實際在練3s但用2s/1s的方式)
"""

import argparse
import os
import shutil


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--root-dir', type=str, default='.', dest='root_dir',
                        help='Root directory to rename all env sight')
    parser.add_argument('-os', '--old-sights', type=str, nargs='+', required=True, dest='old_sights',
                        help='Old sight names to replace')
    print('提醒：sight前的dash不用打。例如-os 1s -ns 3s 實際上是把 -1s 改成 -3s')
    parser.add_argument('-ns', '--new-sight', type=str, required=True, dest='new_sight',
                        help='New sight name to replace with')
    args = parser.parse_args()
    return args


def main(args):
    print(f'Root directory: {args.root_dir}'
          f'\nOld sight names: {args.old_sights}'
          f'\nNew sight name: {args.new_sight}')

    # 遍歷下兩層的目錄
    # (第一層是alg, 第二層是env)
    # 改 env 的名字

    seen_env_names = set()

    for alg in os.listdir(args.root_dir):
        alg_dir = os.path.join(args.root_dir, alg)
        if not os.path.isdir(alg_dir):
            continue

        for env in os.listdir(alg_dir):
            env_dir = os.path.join(alg_dir, env)
            if not os.path.isdir(env_dir):
                continue

            # 改 env 的名字
            found = False
            for old_sight in args.old_sights:
                if f'-{old_sight}' not in env:
                    continue
                new_env = env.replace(f'-{old_sight}', f'-{args.new_sight}')
                new_env_dir = os.path.join(alg_dir, new_env)
                # Rename using shutil.move
                shutil.move(env_dir, new_env_dir)

                print(f'{alg_dir} / {env_dir} -> {new_env_dir}')
                seen_env_names.add(new_env)
                found = True
            if not found:
                seen_env_names.add(env)

    print(f'All env names: {seen_env_names}')


if __name__ == '__main__':
    ARGS = get_args()
    main(ARGS)
