"""
假設你用 search.py　跑完　grid　search　後，你想要找到最好的結果，你可以用這個 script 來找到最好的結果。
輸入：
- file_dir: 你的結果資料夾(底下含有很多 ``1/`` ``2/`` ... 的資料夾)

E.g.,  python tools/get_grid_search_best.py -f results/sacred/iql/lbforaging:Foraging-15x15-3p-5f-v1 --cut-after --n-test 41

"""

import argparse
import json
import os
from collections import defaultdict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file-dir', '--file_dir', type=str,
                        help='The directory containing the results', dest='file_dir', required=True)

    # The variables
    parser.add_argument('--var', nargs='+', type=str, help='The variables to be compared',
                        required=True, dest='var')

    # Data Checking

    # E.g.,
    #       --cut-after 2050000 --n-test 41 (we expect ~2000000, 50000 is a period)
    #                     => after cutting, 41 in total

    parser.add_argument('--cut-after', type=int, help='Cut the test results after this number of steps',
                        required=True, dest='cut_after')
    parser.add_argument('--n-test', type=int, help='Each experiment should have n-test test results',
                        required=True, dest='n_test')

    # Whether to plot (currently only support 2 variables)
    parser.add_argument('--plot', action='store_true', help='Plot the results')

    args = parser.parse_args()

    file_dir = args.file_dir
    variables = args.var
    n_var = len(variables)
    assert n_var > 0, 'No variables to compare'

    # Get all the results
    results = defaultdict(list)
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

        test_return_step_to_mean = {step: mean for step, mean in zip(metric_data['test_return_mean']['steps'],
                                                                     metric_data['test_return_mean']['values'])}

        # Cut
        for step in list(test_return_step_to_mean.keys()):
            if step > args.cut_after:
                test_return_step_to_mean.pop(step)

        assert len(test_return_step_to_mean) == args.n_test, (
            f'{exp_dir} does not have {args.n_test} test results; '
            f'got {len(test_return_step_to_mean)}\n'
            f'; its steps: {test_return_step_to_mean.keys()}')

        max_return = max(metric_data['test_return_mean']['values'])

        combination = []
        for var in variables:
            combination.append(config_data[var])

        results[tuple(combination)].append(max_return)
        print(f'Combination: {combination}, max return: {max_return}')

    # Compute avg over seeds
    cur_max = -float('inf')
    cur_max_dir = None
    cur_max_combination = None
    for combination, returns in results.items():
        results[combination] = sum(returns) / len(returns)
        if results[combination] > cur_max:
            cur_max = results[combination]
            cur_max_combination = combination

    if cur_max is None:
        raise ValueError('No valid results found')

    print(f'Best result: {cur_max}')
    print(f'Best combination: {cur_max_combination}')

    if args.plot:
        import matplotlib.pyplot as plt

        assert n_var in [1, 2, 3]

        if n_var == 1:
            # Each var is an axis
            fig, ax = plt.subplots()
            min_val, max_val = min(results.values()), max(results.values())
            for combination, value in results.items():
                # Use color to represent the value
                # The value for plotting color here is normalized
                color = (value - min_val) / (max_val - min_val)
                print(f'Combination: {combination}, avg value over seeds: {value}, color: {color}')
                # Plot the point
                ax.scatter(combination[0], 0, color=(color, 0, 1 - color))
                # Write the text
                ax.text(combination[0], 0, f'{value:.5f}')
            ax.set_xlabel(variables[0])
            # ax.set_xscale('log')
            plt.show()


        elif n_var == 2:

            # Each var is an axis
            fig, ax = plt.subplots()
            min_val, max_val = min(results.values()), max(results.values())
            for combination, value in results.items():
                # Use color to represent the value
                # The value for plotting color here is normalized
                color = (value - min_val) / (max_val - min_val)
                print(f'Combination: {combination}, avg value over seeds: {value}, color: {color}')
                # Plot the point
                ax.scatter(combination[0], combination[1], color=(color, 0, 1 - color))
                # Write the text
                ax.text(combination[0], combination[1], f'{value:.5f}')
            ax.set_xlabel(variables[0])
            ax.set_ylabel(variables[1])
            plt.show()

        elif n_var == 3:
            # Each var is an axis
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            min_val, max_val = min(results.values()), max(results.values())
            for combination, value in results.items():
                # Use color to represent the value
                # The value for plotting color here is normalized
                color = (value - min_val) / (max_val - min_val)
                print(f'Combination: {combination}, avg value over seeds: {value}, color: {color}')
                # Plot the point
                ax.scatter(combination[0], combination[1], combination[2], color=(color, 0, 1 - color))
                # Write the text
                ax.text(combination[0], combination[1], combination[2], f'{value:.5f}')
            ax.set_xlabel(variables[0])
            ax.set_ylabel(variables[1])
            ax.set_zlabel(variables[2])
            plt.show()
