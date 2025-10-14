import argparse
import json
import os
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

MODEL_NAME_TO_PATH = {
    'evalEpsilon0.05_load10001192_buf20000': 'iql_seed0_lbforaging:Foraging-6s-15x15-3p-5f-v1_2024-08-26 16:35:53.916601_buffer.pkl',
    'evalEpsilon0.2_load10001192_buf20000': 'iql_seed0_lbforaging:Foraging-6s-15x15-3p-5f-v1_2024-08-26 21:02:09.534792_buffer.pkl',
    'evalEpsilon0.5_load10001192_buf20000': 'iql_seed0_lbforaging:Foraging-6s-15x15-3p-5f-v1_2024-08-27 12:32:15.943351_buffer.pkl',
    'evalEpsilon0.2_load5400580_buf20000': 'iql_seed0_lbforaging:Foraging-6s-15x15-3p-5f-v1_2024-08-27 13:00:09.651733_buffer.pkl',
}

MODEL_ABBREVIATION = {
    'StrongModel+LowEpsilon': 'evalEpsilon0.05_load10001192_buf20000',
    'StrongModel+MediumEpsilon': 'evalEpsilon0.2_load10001192_buf20000',
    'StrongModel+HighEpsilon': 'evalEpsilon0.5_load10001192_buf20000',
    'WeakModel+MediumEpsilon': 'evalEpsilon0.2_load5400580_buf20000',
}

OFFLINE_GRAD_STEP_NUMBERS = [5000, 50000, 200000]

SIGHTS = ['1s', '2s', '3s', '4s', '5s', '6s']

SEEDS = [0, 1, 2, 3, 4]

DIR_PATH = '/media/ppo/workspace/epymarl_research/projects/LBF/0827_offline_train_sacred'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, default=DIR_PATH, help='Directory path to the results')
    return parser.parse_known_args()[0]


def get_runs_dir(args, model_abbr, grad_steps, sight):
    env_id = 'lbforaging:Foraging-6s-15x15-3p-5f-v1'
    model_name = MODEL_ABBREVIATION[model_abbr]
    run_dir = os.path.join(args.dir, f'{model_name}_grad{grad_steps}_{sight}', env_id)
    return run_dir


def main(args):
    results = dict(test_return_mean={}, td_loss={})
    for model_abbr, grad_steps, sight in tqdm(
            product(MODEL_ABBREVIATION.keys(), OFFLINE_GRAD_STEP_NUMBERS, SIGHTS), desc='Processing',
            total=len(MODEL_ABBREVIATION) * len(OFFLINE_GRAD_STEP_NUMBERS) * len(SIGHTS)):
        runs_dir = get_runs_dir(args, model_abbr, grad_steps, sight)
        # Get each run's data
        seeds = []
        test_return_means = []
        td_losses = []
        for run_dir in os.listdir(runs_dir):
            if run_dir.startswith('_'):
                continue
            # Get each run's data
            metric_json_path = os.path.join(runs_dir, run_dir, 'metrics.json')
            config_json_path = os.path.join(runs_dir, run_dir, 'config.json')
            # Get seed info
            config_data = json.load(open(config_json_path))
            seed = config_data['seed']
            seeds.append(seed)
            # Get metrics
            metric_data = json.load(open(metric_json_path))
            # Get test return mean
            assert len(metric_data['test_return_mean']['values']) == 1
            test_return_mean = metric_data['test_return_mean']['values'][0]
            test_return_means.append(test_return_mean)
            td_loss = metric_data['offline_train/td_loss']['values'][-1]
            td_losses.append(td_loss)
        # Check seeds duplicate
        problems = []
        if len(seeds) != len(set(seeds)):
            problems.append('f"Seeds duplicate: {seeds}. "')
        # Check lack of seeds
        for seed in SEEDS:
            if seed not in seeds:
                problems.append(f"Seed {seed} is missing.")
        if len(problems) > 0:
            print(f"Problems in {runs_dir}: ")
            for problem in problems:
                print('\t', problem)

        # Aggregate results
        results['test_return_mean'][(model_abbr, grad_steps, sight)] = test_return_means
        results['td_loss'][(model_abbr, grad_steps, sight)] = td_losses
    return results


def plot():
    # Load from pickle
    import pickle
    with open('offline_results.pkl', 'rb') as f:
        results = pickle.load(f)
    # Print out
    for tag, tag_result in results.items():
        print(f'[{tag}]')
        for key, values in tag_result.items():
            print(key, values)

    # 畫 subplots
    # ax row 是 model_abbr; ax column 是 grad steps
    # ax[?][?]: x-axis 是 sight, y-axis 是 tag value
    for tag, tag_result in results.items():
        fig, axs = plt.subplots(len(MODEL_ABBREVIATION), len(OFFLINE_GRAD_STEP_NUMBERS), figsize=(20, 10))
        for i, model_abbr in enumerate(MODEL_ABBREVIATION.keys()):
            for j, grad_steps in enumerate(OFFLINE_GRAD_STEP_NUMBERS):
                ax = axs[i][j]
                ax.set_title(f'{model_abbr} grad{grad_steps}')
                for sight in SIGHTS:
                    key = (model_abbr, grad_steps, sight)
                    values = tag_result[key]
                    mean = sum(values) / len(values)
                    std = np.std(values)
                    ax.scatter(sight, mean, label=sight)
                    ax.errorbar(sight, mean, yerr=std, fmt='o')
                ax.legend()
        plt.tight_layout()
        plt.xlabel('Sight')
        plt.ylabel(tag)
        plt.title(tag)
        plt.show()


    # # 每一個標籤(例如 test_return_mean，td_loss, ...)
    # # 要畫的圖：
    # #       - 固定 model_abbr，畫 grad_steps 對 sight 的 2D 圖
    # #       - 固定 grad_steps, 畫 model_abbr 對 sight 的 2D 圖
    # for tag, tag_result in results.items():
    #     if tag == 'td_loss':
    #         continue
    #     # 每一個 model_abbr 畫 grad_steps 對 sight 的 2D 圖
    #     for model_abbr in MODEL_ABBREVIATION.keys():
    #         plt.figure()
    #         plt.title(f'{tag} of {model_abbr}')
    #         for grad_steps in OFFLINE_GRAD_STEP_NUMBERS:
    #             for sight in SIGHTS:
    #                 key = (model_abbr, grad_steps, sight)
    #                 values = tag_result[key]
    #                 mean = sum(values) / len(values)
    #                 std = np.std(values)
    #                 # 點的顏色代表平均值，越大越紅，越小越藍(0.00~0.0.40)
    #                 # 讓最藍是 0 最紅是 0.40
    #                 # color = np.clip(mean, 0, 0.40)
    #                 color = mean
    #                 # 一般點的值大約在 0.03~0.40，要放大不然看不到
    #                 # 然後要在點上標上平均值
    #                 # sight_number = int(sight[:-1])
    #                 # print(f'sight_number: {sight_number}')
    #                 plt.scatter(str(grad_steps), sight, s=0, c=color, cmap='coolwarm')
    #                 # 文字放在點下面置中
    #                 plt.text(str(grad_steps), sight, f'{mean:.2%}±{std:.2%}', fontsize=10, ha='center', va='center')
    #         # plt.colorbar()
    #         plt.tight_layout()
    #         plt.show()

    return


# plot()

if __name__ == '__main__':
    RESULTS = main(get_args())
    import pickle

    with open('offline_results_.pkl', 'wb') as F:
        pickle.dump(RESULTS, F)
