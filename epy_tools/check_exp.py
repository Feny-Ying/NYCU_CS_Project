"""檢查實驗是否完整的 Script

Sacred 資料夾結構： sacred/ → ${alg+sight} → ${env}/ → ${runs}/
    ${alg+sight} 的結構為：alg name + sight name
    E.g., iql_3s, iql_ucb(1,2,3)

Input:
    - Path to the experiment folder (sacred)
    - Config yaml file path

##### 要檢查的：

- 所有 map, alg, sight, seed 組合都有？
- 多啥？
- 少啥？
- Runs 裡的 seed 是正確的嗎？有重複嗎？有少嗎？
- `test_return_mean` 數量、週期是正確的嗎？

##### 執行此檢查程式，需要給定在 config：

- Algorithms
- Map names x Sight settings
- Seed 數字有哪些
- test_interval, min_n_tests

#### 資料夾結構：

假設資料夾結構如下：`sacred/` → `${alg+sight}` → `${env}/` → `${runs}/`
    - 第 1 層：method1, method2, method3, ... (這裡的 method 指的是 alg+sight)
    - 第 2 層：env1, env2, env3, ...
    - 第 3 層：run1, run2, run3, ... (run的編號不一定是seed，順序可能是亂的); 這層略過名字前面有 `_` 的資料夾

#### Runs 的檢查

資料來源：
- `config.json`：裡面有 seed 和其它設定 -> 這裡檢查是否 test_interval 正確 以及抓出其 seed
- `metrics.json`：裡面有 `test_return_mean` 的數值 ->
    - 檢查是否有正確的數量
檢查後就可以為該 run 建一個 class instance object

### 結果

首先秀出「空資料夾」(空 alg+sight, env, run 資料夾)，若有空資料夾，直接 raise error。

秀出所有"多的"組合。
- 多的 map
- 多的 alg+sight
- 多的 seed

最後，畫一個 summary table，秀出哪些方法的哪些環境的哪些seed還沒有完成。
Row0: alg name
Row1: sight name
Columns: map names
Cell: 該有的各seed完成了嗎？若沒完成是有什麼問題？(長度不足？interval有誤？)
"""
import argparse
import json
import os
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pprint import pprint
from typing import Optional, Iterable

import pandas as pd
import yaml
from tqdm import tqdm


@dataclass
class Run:
    """
    alg+sight string: 用它的路徑來看
    map env string: 用它的路徑來看

    - 檢查是否有正確的數量
    """
    # Basic data
    path: str
    method_name: str
    alg_name: str
    sight_name: str
    map_name: str
    # Get from the `config.json`
    seed: int
    test_interval: int
    # Get from the `metrics.json`
    test_return_means: list[float]
    test_steps: list[int]
    # Check whether the data meets the requirements
    valid_test_interval: bool
    valid_test_return_mean_numbers: bool
    # Hyperparameter and problems
    config: dict
    hp_problems: list[str]


def recurve_update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = recurve_update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def get_map_id_from_config(config, map_name, sight) -> str:
    return config['map_to_sight_to_sight_and_env_id'][map_name][sight]['env_id']


def get_from_yaml_config(config, alg, key) -> str:
    alg_config = config['possible_algorithms'][alg]
    try:
        v_all = config['all_shared_hyperparameters'][key]
        if len(v_all) != 1:
            raise ValueError(f'Key {key} in all_shared_hyperparameters should only have one value. '
                             f'Please check the config file.')
    except KeyError:
        v_all = None
    try:
        v_alg = config['hyperparameter_groups'][alg_config['hyperparameter_group']][key]
        if len(v_alg) != 1:
            raise ValueError(f'Key {key} in hyperparameter_groups should only have one value. '
                             f'Please check the config file.')
        if v_all and v_alg:
            msg = 'Both all_shared_hyperparameters and hyperparameter_groups have the same key.'
            if v_all != v_alg:
                msg += 'The key in all_shared_hyperparameters and hyperparameter_groups are different.'
            raise ValueError(msg)
    except KeyError:
        v_alg = None
    if v_all is None and v_alg is None:
        raise ValueError(f'Key {key} not found in the config. Please check the config file.')
    v = v_all or v_alg
    return v[0]


def compute_min_t_max(test_interval, min_n_tests, time_limit, batch_size_run):
    return (test_interval * (min_n_tests - 1)) + (time_limit * batch_size_run * min_n_tests)


def make_cmd(alg, sight, map_name, seed, config):
    """
    example:
    ```
    CUDA_VISIBLE_DEVICES=0 python3 main.py --config="lbf/ippo/base" \n
    --env-config=gymma with env_args.time_limit=50  \n
    env_args.key="lbforaging:Foraging-3s-15x15-4p-5f-coop-v1" seed=0 t_max=20050000  \n
    save_path="result_0801_lbf_ppo"
    ```

    """
    cmd_conf = config['make_cmd']
    yaml_str = cmd_conf['yaml']
    yaml_domain = yaml_str.split('/')[0]
    date_str = f'{datetime.now():%Y%m%d}'
    save_path = f'result_{yaml_domain}_{date_str}'
    name = f'{alg}_{sight}'
    preprocess_desc = cmd_conf['preprocess_desc_dict'].get(sight, None)
    alg_config = config['possible_algorithms'][alg]
    time_limit = cmd_conf['time_limit']
    min_n_tests = alg_config['min_n_tests']
    batch_size_run = get_from_yaml_config(config, alg, 'batch_size_run')
    test_interval = get_from_yaml_config(config, alg, 'test_interval')
    env_id = get_map_id_from_config(config, map_name, sight)

    # Auto computed t_max
    min_t_max = compute_min_t_max(test_interval, min_n_tests, time_limit, batch_size_run)

    cmd = (f'CUDA_VISIBLE_DEVICES= python3 main.py --config="{yaml_str}" '
           f'--env-config=gymma with env_args.time_limit={time_limit} '
           f'env_args.key="{env_id}" seed={seed} t_max={min_t_max} '
           f'save_path="{save_path}" name="{name}" preprocess_desc="{preprocess_desc}" ')

    # # ------------ For special hyperparameters ------------
    # # 開啟 default.yaml env裡 gymma yaml 和 alg yaml (`yaml_str`) 的 hyperparameters，組成一個 dict
    # # 再用 config 裡面的 hyperparameters 來看有什麼是和 dict 裡面不一樣的額外給在 cmd
    #
    # # 讀 default.yaml
    # default_yaml_config = yaml.safe_load(open('src/config/default.yaml', 'r'))
    # # 讀 gymma env yaml
    # env_yaml_config = yaml.safe_load(open(f'src/config/envs/gymma.yaml', 'r'))
    # # 讀 alg yaml
    # alg_yaml_config = yaml.safe_load(open(f'src/config/algs/{yaml_str}', 'r'))
    #
    # # 組成一個 dict (後者會覆蓋前者)
    # # Recursively update the dict
    # total_yaml_config = recurve_update_dict(default_yaml_config, env_yaml_config)
    # total_yaml_config = recurve_update_dict(total_yaml_config, alg_yaml_config)

    for k, v in cmd_conf['custom_hyperparameters'].items():
        if isinstance(v, str):
            cmd += f'{k}="{v}" '
        else:
            cmd += f'{k}={v} '

    return cmd


def get_alg_sight(method_name):
    """Use the first `_` to split the method name into alg and sight names.
    If there are more than one `_`, combine the rest back."""
    parts = method_name.split('_')
    alg = parts[0]
    sight = '_'.join(parts[1:])
    return alg, sight


def get_key(d, val) -> Optional[str]:
    """Get the key of a value in a dict. Return None if not found"""
    return next((key for key, value in d.items() if value == val), None)


def get_map_by_sight_and_env_id(sight, env_id, sight_to_env_id_to_map) -> Optional[str]:
    if sight not in sight_to_env_id_to_map:
        return None
    if env_id not in sight_to_env_id_to_map[sight]:
        return None
    return sight_to_env_id_to_map[sight][env_id]


def get_args():
    parser = argparse.ArgumentParser()
    # Required
    parser.add_argument('-f', '--folder', type=str, help='Path to the Sacred folder',
                        default='../projects/POEM/sacred')
    parser.add_argument('-c', '--config', type=str, help='Path to the config yaml file',
                        default='check_exp_config/lbf.yml')
    # Optional
    parser.add_argument('-v', '--verbose', action='store_true', help='Print verbose information')
    parser.add_argument('--skip-alg', nargs='+', help='Skip checking these alg names', default=[])
    # Useful tools
    parser.add_argument('-del', '--delete', action='store_true', help='Delete the invalid runs')
    parser.add_argument('-cmd', '--make-cmd', action='store_true',
                        help='Output commands to run the runs that are not lacking!')
    return parser.parse_args()


def check_run_valid(seed, run, each_info: bool = True, simplified_path: bool = False,
                    force_show_info=False) -> [str, bool]:
    """檢查 seed 的 valid 狀態，並回傳訊息"""
    msg = ''
    if run.path:
        path_str = f'path={run.path} ' if not simplified_path else f'path={run.path.split("/")[-1]}'
    else:
        path_str = ''
    tmp_msg = ''
    if not run.valid_test_interval:
        tmp_msg += f'test_interval not valid'
    if not run.valid_test_return_mean_numbers:
        if len(tmp_msg) > 0:
            tmp_msg += ', '
        tmp_msg += f'test_return_mean_numbers not valid'
    if len(tmp_msg) > 0 or force_show_info:  # Invalid
        msg = f'{path_str} seed={seed} {tmp_msg}'
        if each_info:
            last_step = run.test_steps[-1] if run.test_steps else 'None'
            msg += f' (n_tests={len(run.test_steps)}, last_step={last_step}) '
        msg += '\n'
    return msg, len(tmp_msg) == 0


def main():
    args = get_args()

    # Load yaml config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        if args.verbose:
            print('Loaded config:')
            pprint(config)

    if args.make_cmd:
        raise NotImplementedError('This feature is not supported yet.')
        warnings.warn('Make_cmd does ONLY support gymma env (Note that SMAC is not supported so far).')

    # -------------------- 超參數資料建置 --------------------

    # 先把超參數的部分讀出來整理好，後面要用 (這裡 "hp" 是 hyperparameter 的縮寫)
    all_shared: dict[str, list] = config['all_shared_hyperparameters']  # 記錄在 config 裡面此domain所有alg shared 的 hp
    all_skip_list: list[str] = config['all_skipped_hyperparameters']  # 記錄在 config 裡面此domain所有alg跳過檢查的 hp
    alg_not_having_ok_hyperparameters: dict[str, list[str]] = {}  # 記錄在 config 裡面此domain所有alg不需要檢查的 hp
    hp_groups: dict[str, dict[str, list]] = config['hyperparameter_groups']  # 記錄在 config 裡面的 hp group 設定
    alg_hyperparameters: dict[str, dict[str, list]] = {}  # alg_name -> hyperparameter_name -> list[allowed values]
    for alg_name in config['possible_algorithms'].keys():
        # Use the base hyperparameters
        hp_dict = deepcopy(all_shared)
        # If given group, use group hyper param as base
        hp_group: Optional[str] = config['possible_algorithms'][alg_name]['hyperparameter_group']
        if hp_group:
            for hy_name, allowed_values in hp_groups[hp_group].items():
                hp_dict[hy_name] = allowed_values
        # If given custom hyper param, update the base
        for hy_name, allowed_values in config['possible_algorithms'][alg_name]['custom_hyperparameters'].items():
            hp_dict[hy_name] = allowed_values
        alg_hyperparameters[alg_name] = hp_dict
        alg_not_having_ok_hyperparameters[alg_name] = config['possible_algorithms'][alg_name][
            'not_having_ok_hyperparameters']

    # 先建好 dict: sight -> (env_id, preprocess_desc) -> map_name
    sight_to_env_id_to_map: dict[str, dict[tuple[str, str], str]] = {}
    for map_name, map_data in config['map_to_sight_to_sight_and_env_id'].items():
        for sight, sight_data in map_data.items():
            env_id = sight_data['env_id']
            sight_to_env_id_to_map.setdefault(sight, {})[env_id] = map_name

    # -------------------- 資料夾挖掘 --------------------

    # 有了 config 資料後，我們就可以建出整個巢狀的 dict。待會深挖資料夾時，有找到的就填入。
    # Dict 結構： map_name -> alg_name -> sight_name -> seed -> Value: list[Run] (原則上list長度為1，若有多個，代表有問題)
    # 這裡先建立一個空的 dict
    all_runs: dict[str, dict[str, dict[str, dict[int, list[Run]]]]] = {}
    for map_name in config['maps'].keys():
        all_runs[map_name] = {}
        for alg in config['possible_algorithms'].keys():
            all_runs[map_name][alg] = {}
            for sight in config['maps'][map_name]['sights']:
                all_runs[map_name][alg][sight] = {}
                for seed in config['seeds']:
                    all_runs[map_name][alg][sight][seed] = []

    # 一大串 dict: map_name -> alg_name -> sight_name -> seed
    # 這裡做：讀出 folder 下的所有資料夾內的 runs。
    # 這個過程若有空資料夾，直接 raise error。
    # 這個過程中，把所有的 runs 用 Run class instance object 來表示。

    # TODO: 挖的過程，若有 alg 不在 config['possible_algorithms'].keys() 裡面，直接 raise error。

    # Get method names (method here = alg+sight)
    method_names = [d for d in os.listdir(args.folder) if os.path.isdir(os.path.join(args.folder, d))]
    method_names = sorted(method_names)
    if args.verbose:
        print(f'Founded method names:')
        pprint(method_names)

    # 用來記錄哪些 method 有 env 是多的
    extra_methods_contain_no_wanted_envs: list[str] = []
    extra_method_envs = defaultdict(list)
    extra_method_names: list[str] = []
    extra_seed_runs: list[Run] = []

    # Get total number of method_name x env_name numbers
    total_method_env_numbers = 0
    for method_name in method_names:
        method_dir = os.path.join(args.folder, method_name)
        total_method_env_numbers += len(os.listdir(method_dir))

    # tqdm
    pbar = tqdm(total=total_method_env_numbers, desc='Checking methods x envs')

    # Check each method's envs and seeds
    config_json_not_found_hp_dict: dict[str, list[str]] = defaultdict(list)  # key: hp name, value: list of paths
    for method_name in method_names:
        method_dir = os.path.join(args.folder, method_name)

        # Get alg, sight names
        try:
            alg, sight = get_alg_sight(method_name)
        except ValueError:
            raise ValueError(f'Invalid method name: {method_name}')

        # --------- Check envs ---------

        # If method folder is empty, raise error
        if len(os.listdir(method_dir)) == 0:
            raise ValueError(f'Empty env names for {method_name}')

        # 註：這裡的 env_name 主要是資料夾名稱、地圖環境名全稱，而 map_name 是 config 裡面的簡稱
        env_names = []
        for env_name in os.listdir(method_dir):
            # Check if this env id is in the config
            # Get the key of the map name in the config
            map_name = get_map_by_sight_and_env_id(sight, env_name, sight_to_env_id_to_map)

            if map_name is None:
                extra_method_envs[method_name].append(env_name)
                continue

            env_names.append(env_name)

        if args.verbose:
            print(f'Founded env names for {method_name}: {env_names}')

        if len(env_names) == 0:
            extra_methods_contain_no_wanted_envs.append(method_name)

        if alg in args.skip_alg:
            # extra_method_names.append(method_name)
            pbar.update(len(env_names))
            continue

        for env_name in env_names:
            env_dir = os.path.join(method_dir, env_name)

            # Get map name
            map_name = get_map_by_sight_and_env_id(sight, env_name, sight_to_env_id_to_map)
            assert map_name is not None, f'env_name {env_name} not in config, 這在前面應已檢查過了'

            # Check if the alg & sight are in this map's settings
            if alg not in config['maps'][map_name]['algorithms'] or sight not in config['maps'][map_name]['sights']:
                extra_method_names.append(method_name)
                pbar.update(1)
                continue

            # --------- Check runs ---------

            # Skip the runs with `_`. E.g., `_sources`, which is produced automatically, and it is not a run
            run_names = [d for d in os.listdir(env_dir) if
                         os.path.isdir(os.path.join(env_dir, d)) and not d.startswith('_')]

            # Sort the run names by str length and str alphabetically
            run_names = sorted(run_names, key=lambda x: (len(x), x))
            if len(run_names) == 0:
                warnings.warn(f'Empty run names for {method_name}/{env_name}')
                if args.delete:
                    os.rmdir(env_dir)
                    print(f'Deleted empty folder: {env_dir}')

            if args.verbose:
                print(f'Founded run names for {method_name}/{env_name}:')
                # pprint(run_names)

            for run_name in run_names:
                run_dir = os.path.join(env_dir, run_name)

                # Load config and metric data
                config_path = os.path.join(run_dir, 'config.json')
                if not os.path.exists(config_path):
                    raise FileNotFoundError(f'config.json not found in {run_dir}')
                metric_path = os.path.join(run_dir, 'metrics.json')
                if not os.path.exists(metric_path):
                    raise FileNotFoundError(f'metrics.json not found in {run_dir}')
                config_data = json.load(open(config_path, 'r'))
                metric_data = json.load(open(metric_path, 'r'))

                if args.verbose:
                    print(f'Checking config & metrics for {method_name}/{env_name}/{run_name}:')

                # Get data
                seed = config_data['seed']
                test_interval_configured = config_data['test_interval']
                # test_interval_configured = get_from_yaml_config(config, alg, 'test_interval')

                hp_problems: list[str] = []  # a list of problem messages

                # metric.json `test_return_means`'s dict_keys(['steps', 'timestamps', 'values'])
                # if 'test_return_mean' not in metric_data.keys():
                #     raise ValueError(f'test_return_mean not found in "{run_dir}"')
                metric_test_return_dict = metric_data.get('test_return_mean', None)
                if metric_test_return_dict is None:
                    hp_problems.append('test_return_mean not found in metric.json.')
                    metric_test_return_dict = {'values': [], 'steps': []}

                test_return_means = metric_test_return_dict.get('values', None)
                if test_return_means is None:
                    hp_problems.append('values not found in metric.json test_return_mean dict.')
                    test_return_means = []
                test_steps = metric_test_return_dict.get('steps', None)
                if test_steps is None:
                    hp_problems.append('steps not found in metric.json test_return_mean dict.')
                    test_steps = []

                # Whether valid
                assert len(alg_hyperparameters[alg]['test_interval']) == 1, 'test_interval should only have one value.'
                valid_configured_test_interval = test_interval_configured == alg_hyperparameters[alg]['test_interval'][0]
                if args.verbose:
                    print(f'n_tests={len(test_return_means)}')
                valid_empirical_test_return_numbers = len(
                    test_return_means) >= config['possible_algorithms'][alg]['min_n_tests']  # Must be >=

                # Check if all hp meet our requirements; if not, list all the problems
                not_found_hp_messages: list[str] = []  # a list of hp names not found in the config
                for hp_name, hp_v in config_data.items():  # 檢查有在config.json裡面的超參是否符合config yaml的設定
                    if hp_name in all_skip_list:  # 有註記的跳過
                        continue
                    if hp_name in alg_hyperparameters[alg].keys():
                        # # check if iterables
                        if not isinstance(alg_hyperparameters[alg][hp_name], Iterable):
                            raise ValueError(
                                f'the value of {hp_name}: {alg_hyperparameters[alg][hp_name]} is not iterable. '
                                f'Please check the config file.')
                        if hp_v not in alg_hyperparameters[alg][hp_name]:
                            hp_problems.append(
                                f'{hp_name}={hp_v} not in allowed values: {alg_hyperparameters[alg][hp_name]}')
                    else:
                        not_found_hp_messages.append(f'Found unknown hyperparameter: {hp_name} = {hp_v} in {run_dir}. '
                                                     f'Please add it to the config file (shared/skipped/not_having).')

                # Check "env_id" and "preprocess_desc" in config.json
                # If there is no key "key", skip this check and check "map_name" (e.g., SMAC doesn't use env_args.key)
                if config_data['env_args'].get('map_name', None):
                    if config_data['env_args']['map_name'] != env_name:
                        hp_problems.append(f'env_args.map_name={config_data["env_args"]["map_name"]} != {env_name}')
                else:
                    if config_data['env_args']['key'] != env_name:
                        hp_problems.append(f'env_args.key={config_data["env_args"]["key"]} != {env_name}')

                # TODO: 目前的check機制對於SC2, 若有 ucb 從>1個選會有 sightInfo，若==1則其為False。這個詳細設定檢查不到，所以要人工注意！

                # 這裡反過來看有config的超參是否都有在run的config裡面
                for hp_name in alg_hyperparameters[alg].keys():
                    if hp_name not in config_data.keys():
                        if hp_name in alg_not_having_ok_hyperparameters[alg]:
                            continue
                        config_json_not_found_hp_dict[hp_name].append(run_dir)
                        # not_found_hp_messages.append()

                # if len(not_found_hp_messages) > 0:
                #     raise ValueError('\n'.join(not_found_hp_messages))

                # Construct a Run object
                run = Run(path=run_dir,
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
                          test_steps=test_steps)

                # Fill the data into the all_runs dict
                if seed in all_runs[map_name][alg][sight].keys():
                    all_runs[map_name][alg][sight][seed].append(run)
                else:
                    extra_seed_runs.append(run)

            pbar.update(1)

    pbar.close()

    # If not found hp in config.json, raise error
    if len(config_json_not_found_hp_dict) > 0:
        print('========= Not found hyperparameters in config.json =========')
        pprint(config_json_not_found_hp_dict)
        raise ValueError('Some hyperparameters are not found in config.json. Please check the config file.')

    extra_file_summary = []

    # Print extra methods
    if len(extra_methods_contain_no_wanted_envs) > 0:
        print('========= Extra methods contain no wanted envs =========')
        pprint(extra_methods_contain_no_wanted_envs)
        extra_file_summary.append(
            f'Number of extra methods contain no wanted envs: {len(extra_methods_contain_no_wanted_envs)}')

    if len(extra_method_envs) > 0:
        print('========= Extra method envs =========')
        pprint(extra_method_envs)
        extra_file_summary.append(f'Number of extra method envs: {len(extra_method_envs)}')

    if len(extra_method_names) > 0:
        print('========= Extra method names =========')
        pprint(extra_method_names)
        extra_file_summary.append(f'Number of extra method names: {len(extra_method_names)}')

    if len(extra_seed_runs) > 0:
        print('========= Extra seed runs =========')
        for run in extra_seed_runs:
            print(f'{run.method_name} {run.map_name} {run.seed} {run.path}')
        extra_file_summary.append(f'Number of extra seed runs: {len(extra_seed_runs)}')

    # ------------ 最後，畫一個 summary table，秀出哪些方法的哪些環境的哪些seed還沒有完成 ------------
    # Row0: alg name
    # Row1: sight name
    # Columns: map names
    # Cell: 該有的各seed完成了嗎？若沒完成是有什麼問題？(長度不足？有什麼valid的問題？重複seed？)

    # Create an empty DataFrame
    # Each row is an alg x sight combination
    # Each column is a map name
    # Cell: details of the seeds, whether they are done or not

    # index = []
    # for alg in config['possible_algorithms'].keys():
    #     for sight in all_runs[alg].keys():
    #         index.append(f'{alg}_{sight}')
    # summary_table = pd.DataFrame(index=index, columns=config['maps'].keys())

    col_alg = []
    col_sight = []
    col_map = []
    col_msg = []
    commands = []

    # Loop through all_runs to fill the summary table
    all_done = True
    n_done = 0
    n_not_yet = 0
    for map_name in all_runs.keys():
        for alg in all_runs[map_name].keys():
            if args.skip_alg:
                if alg in args.skip_alg:
                    continue
            for sight in all_runs[map_name][alg].keys():
                # ---------- Make message ----------
                msg = ''
                invalid_runs: list[Run] = []
                invalid_run_messages: list[str] = []
                not_finished_seeds = set(config['seeds'])
                for seed in all_runs[map_name][alg][sight].keys():
                    seed_runs = all_runs[map_name][alg][sight][seed]

                    # Add hp problem msg
                    for run in seed_runs:
                        if len(run.hp_problems) > 0:
                            hp_problem_total_str = "\n".join(run.hp_problems)
                            run_msg = hp_problem_total_str
                            invalid_runs.append(run)
                            invalid_run_messages.append(run_msg)
                            msg += f'{seed} path={run.path} hp_problems: \n{hp_problem_total_str}; \n'

                    # Check runs problems
                    if len(seed_runs) == 0:
                        msg += f'Seed={seed} not found; \n'
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
                        duplicated_desc_end = ": \n" if len(tmp_msg) > 0 else "."
                        msg += f'Seed={seed} duplicated{duplicated_desc_end}'
                        msg += tmp_msg
                        msg += '\n'

                # Automatically delete invalid runs
                if args.delete:
                    for run, run_msg in zip(invalid_runs, invalid_run_messages):
                        print(
                            f'[Invalid Run] alg={run.alg_name} map={run.map_name} sight={run.sight_name} '
                            f'seed={run.seed} path={run.path}\n'
                            f'Message: {run_msg}')
                        response = input('Delete this invalid run? (y/n): ')
                        if response == 'y':
                            try:
                                import shutil
                                shutil.rmtree(run.path)
                                print(f'Deleted invalid run: {run.path}')
                                msg += f'Invalid run {run.path} deleted;\n'
                            except Exception as e:
                                print(f'Error deleting invalid run: {run.path}')
                                print(e)
                        else:
                            print(f'Not deleting invalid run: {run.path}')
                        print()

                # Make cmd
                if args.make_cmd:
                    commands.append(
                        f'# ------ [MAKE CMD] alg={alg} map={map_name} sight={sight} seeds={not_finished_seeds}\n')
                    # print(f'[MAKE CMD] alg={alg} map={map_name} sight={sight} seeds={not_finished_seeds}\n')
                    print(f'not_finished_seeds: {not_finished_seeds}')
                    for seed in not_finished_seeds:
                        cmd = make_cmd(alg=alg, map_name=map_name, sight=sight, seed=seed, config=config)
                        print(f'cmd = {cmd}')
                        commands.append(cmd)
                        # print(cmd)
                    # print()
                    commands.append('\n')

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
                col_msg.append(msg)

    # Create the DataFrame from the lists
    summary_table = pd.DataFrame({
        'alg': col_alg,
        'sight': col_sight,
        'map': col_map,
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
    # def print_with_newlines(df):
    #     for index, row in df.iterrows():
    #         for col in df.columns:
    #             print(row[col].replace('\n', '\t'), end=' ')
    #         print()
    #
    # print_with_newlines(summary_table)
    # print(summary_table.replace('\\n','\t'))

    # Also save as a csv file
    time_str = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    summary_table_dir = 'exp_summary_table_history'
    os.makedirs(summary_table_dir, exist_ok=True)
    summary_table.to_csv(os.path.join(summary_table_dir, 'check_exp_summary_table_' + time_str + '.csv'), index=True)

    # Show successful message if all methods are done
    if all_done:
        print('★★★★★ All methods are done! ★★★★★')
    else:
        print('★★★★★ NOT all methods are done! ★★★★★')

    # Print extra file summary
    if len(extra_file_summary) > 0:
        print('========= Extra file summary =========')
        print('\n'.join(extra_file_summary))

    if args.make_cmd:
        # print('========= Commands =========')
        # print('\n'.join(commands))
        # Output to a file
        time_str = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        cmd_dir = 'exp_cmd/auto_generated_cmds'
        os.makedirs(cmd_dir, exist_ok=True)
        with open(os.path.join(cmd_dir, f'{time_str}.sh'), 'w') as f:
            f.write('\n'.join(commands))

    print('')
    print(f'Number of done: {n_done} runs, not yet: {n_not_yet} runs, total: {n_done + n_not_yet} runs.')

if __name__ == '__main__':
    main()
