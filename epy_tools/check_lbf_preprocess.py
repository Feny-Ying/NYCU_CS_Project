from typing import Optional

# noinspection PyUnresolvedReferences
import gym
import numpy as np
from tqdm import tqdm

from epy_tools.lbf_utils import pprint_obs, lbf_preprocess


def check_obs_the_same(obs1: list[np.ndarray], obs2: list[np.ndarray]) -> bool:
    """檢查兩個obs是否一樣(list of agents' obs array)"""
    if len(obs1) != len(obs2):
        return False
    for i in range(len(obs1)):
        # noinspection PyUnresolvedReferences
        if not (obs1[i] == obs2[i]).all():
            return False
    return True


def get_obs(sight, seed, map_size, n_agents, n_fruits, return_render=False):
    env = gym.make(f'lbforaging:Foraging-{sight}s-{map_size}x{map_size}-{n_agents}p-{n_fruits}f-v1')
    env.seed(seed)
    obs = env.reset()
    render_result = env.render(mode="rgb_array") if return_render else None
    env.close()
    if return_render:
        return obs, render_result
    return obs


def check_diff_sight_obs(env_,
                         n_agents: int,
                         n_fruits: int,
                         map_size: int,
                         seed: int):
    """給一組環境和原始map size，
    嚐試不同的 original_sight (<= map_size) 和 new_sight (1~original_sight)，
    都測看看preprocess_func和環境給的正解是否一致
    """

    # def get_obs(env__, twice=False):
    #     """非常奇怪的是用同一個環境下一次要seed reset 2 次才會一樣。
    #     或是直接用同一seed直接make新的"""
    #     env__.seed(seed)
    #     obs__ = env_.reset()
    #     render_result = env_.render(mode="rgb_array")
    #     # time.sleep(5)
    #     if twice:
    #         assert not twice
    #         env__.seed(seed)
    #         obs__ = env_.reset()
    #         env_.render()
    #     return obs__, render_result

    # 假設同seed出來的地圖是一樣，即使sight不同。

    # 拿 1~map_size 的 obs 正解
    sight_to_correct_obs = {}
    for sight in range(1, map_size + 1):
        sight_to_correct_obs[sight] = get_obs(sight, seed, map_size, n_agents, n_fruits)

    # 開始嘗試從不同的 original_sight 轉到不同的 new_sight
    for original_sight in range(map_size, 0, -1):
        for new_sight in range(original_sight, 0, -1):
            original_obs = sight_to_correct_obs[original_sight]
            answer_obs = sight_to_correct_obs[new_sight]
            # 拿 preprocess_func 的結果
            preprocessed_obs = lbf_preprocess(
                original_sight=original_sight,
                new_sight=new_sight,
                original_obs=original_obs,
                n_agents=n_agents,
            )
            if not check_obs_the_same(answer_obs, preprocessed_obs):
                print(f'================================= sight {new_sight} ===================')
                print(f'original_sight: {original_sight}, new_sight: {new_sight}, map_size: {map_size}, '
                      f'n_agents: {n_agents}, n_fruits: {n_fruits}, seed: {seed}')
                print(f'########## original ##########')
                pprint_obs(original_obs, n_agents=n_agents, n_fruits=n_fruits)
                print(f'########## correct ##########')
                pprint_obs(answer_obs, n_agents=n_agents, n_fruits=n_fruits)
                print(f'########## preprocess ##########')
                pprint_obs(preprocessed_obs, n_agents=n_agents, n_fruits=n_fruits)
                # Plot
                _, render_result = get_obs(sight=original_sight, seed=seed, map_size=map_size, n_agents=n_agents,
                                           n_fruits=n_fruits, return_render=True)
                import matplotlib.pyplot as plt
                plt.imshow(render_result)
                plt.title(f'original_sight: {original_sight} new_sight: {new_sight}')
                plt.show()

                raise ValueError('preprocess_func is wrong')


#
# # For each original_sight
# for original_sight in range(map_size, 0, -1):
#     # Make base environment
#     env = gym.make(f'lbforaging:Foraging-{original_sight}s-{map_size}x{map_size}-{n_agents}p-{n_fruits}f-v1')
#     # Get original obs in this original_sight
#     original_obs = get_obs(env)
#     # 試不同的sight
#     for new_sight in range(original_sight, 0, -1):
#         # 拿正確答案
#         env_.env.sight = new_sight
#         answer_obs = get_obs(
#             env__=gym.make(f'lbforaging:Foraging-{new_sight}s-{map_size}x{map_size}-{n_agents}p-{n_fruits}f-v1')
#         )
#         # answer_obs = get_obs(env_, twice=True)
#         # 拿 preprocess_func 的結果
#         preprocessed_obs = preprocess_obs(
#             preprocess_desc=f'lbf:{original_sight}s->{new_sight}s',
#             n_agents=n_agents,
#             original_obs=original_obs,
#             verbose=True)
#         print(f'================================= sight {new_sight} ===================')
#         if not check_obs_the_same(answer_obs, preprocessed_obs):
#             print(f'original_sight: {original_sight}, new_sight: {new_sight}, map_size: {map_size}, '
#                   f'n_agents: {n_agents}, n_fruits: {n_fruits}, seed: {seed}')
#             print(f'########## original ##########')
#             pprint_obs(original_obs, n_agents=n_agents, n_fruits=n_fruits)
#             print(f'########## correct ##########')
#             pprint_obs(answer_obs, n_agents=n_agents, n_fruits=n_fruits)
#             print(f'########## preprocess ##########')
#             pprint_obs(preprocessed_obs, n_agents=n_agents, n_fruits=n_fruits)
#             raise ValueError('preprocess_func is wrong')
# env.close()


def check_a_range(agent_range: tuple[int, int],
                  fruit_range: tuple[int, int],
                  map_size_range: tuple[int, int],
                  seed_range: tuple[int, int],
                  desc: Optional[str] = None):
    # Use tqdm to show progress
    total_iterations = (agent_range[1] - agent_range[0] + 1) * (fruit_range[1] - fruit_range[0] + 1) * (
            map_size_range[1] - map_size_range[0] + 1) * (seed_range[1] - seed_range[0] + 1)
    with tqdm(total=total_iterations, desc=desc) as pbar:
        for n_agents in range(agent_range[0], agent_range[1] + 1):
            for n_fruits in range(fruit_range[0], fruit_range[1] + 1):
                for map_size in range(map_size_range[0], map_size_range[1] + 1):
                    for seed in range(seed_range[0], seed_range[1] + 1):
                        check_diff_sight_obs(
                            env_=gym.make(f'lbforaging:Foraging-2s-{map_size}x{map_size}-{n_agents}p-{n_fruits}f-v1'),
                            n_agents=n_agents,
                            n_fruits=n_fruits,
                            map_size=map_size,
                            seed=seed)
                        pbar.update()


def main():
    # Part 1: Small map
    #              - agents: <=3
    #              - fruits: <=3
    #              - map_size: 5~15
    #              - seed: 0~999
    check_a_range(desc='Part 1: Small map',
                  agent_range=(2, 3),
                  fruit_range=(1, 3),
                  map_size_range=(5, 15),
                  seed_range=(0, 999))
    # Part 2: Larger map
    #              - agents: 3~5
    #              - fruits: 3~5
    #              - map_size: 15~20
    #              - seed: 0~999
    check_a_range(desc='Part 2: Larger map',
                  agent_range=(3, 5),
                  fruit_range=(3, 5),
                  map_size_range=(15, 19),
                  seed_range=(0, 999))


if __name__ == '__main__':
    main()
