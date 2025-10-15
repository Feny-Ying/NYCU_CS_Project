from typing import Optional, List

import numpy as np


def pprint_obs(obs, n_agents, n_fruits):
    """清楚的打印各agent的obs (EpisodeBatch 的 obs)"""
    print(f'共有 {n_fruits} 個水果和 {n_agents} 個 agent')
    for i_ag, ag_obs in enumerate(obs):
        print(f'Agent {i_ag}:')
        # for 迴圈一次印 3 個
        for i in range(0, len(ag_obs), 3):
            type_name = 'agent' if i >= n_fruits * 3 else 'fruit'
            no = (i // 3) - 1 if i >= n_fruits * 3 else i // 3
            print(f'>>> {type_name} {no:>2}: ', end='')
            if ag_obs[i] == -1 and ag_obs[i + 1] == -1:
                print('------')
            else:
                print(f'(y,x)=({int(ag_obs[i]):>3},{int(ag_obs[i + 1]):>3})', end=' ')
                print(f'level={int(ag_obs[i + 2])}')


def pprint_se_obs(obs, n_agents, n_fruits):
    """清楚的打印各agent的obs (EpisodeBatch 的 obs)"""
    print(f'共有 {n_fruits} 個水果和 {n_agents} 個 agent')
    for i_ag, ag_obs in enumerate(obs):
        print(f'Agent {i_ag}:')
        # for 迴圈一次印 3 個
        for i in range(0, len(ag_obs) - 1, 3):
            type_name = 'agent' if i >= n_fruits * 3 else 'fruit'
            # no = (i // 3) - 1 if i >= n_fruits * 3 else i // 3
            no = i // 3
            no -= n_fruits if i >= n_fruits * 3 else 0
            print(f'>>> {type_name} {no:>2}: ', end='')
            if ag_obs[i] == -1 and ag_obs[i + 1] == -1:
                print('------')
            else:
                print(f'(y,x)=({int(ag_obs[i]):>3},{int(ag_obs[i + 1]):>3})', end=' ')
                print(f'level={int(ag_obs[i + 2])}')
    print(f'--> cur select sight: {int(ag_obs[-1])}')


def lbf_preprocess(original_sight: int, new_sight: int, original_obs: List[np.ndarray],
                   n_agents: int, n_fruits: Optional[int] = None, verbose: bool = False) -> List[np.ndarray]:
    """Preprocess 1-step lbf obs: a list of each agent's obs array;
    len of each obs array is 3 * (n_fruits + n_agents)

    LBF preprocess cmd format:
    - "15s->10s": change sight from 15 to 10

    """
    if n_fruits is None:
        n_fruits = int(len(original_obs[0]) // 3) - n_agents

    old_sight = original_sight
    if old_sight == new_sight:
        return original_obs

    if verbose:
        print(f'old_sight: {old_sight}, new_sight: {new_sight}')

    # 設定形狀
    obs_shape = original_obs[0].shape
    num_obs = len(original_obs)
    all_obs = np.array(original_obs)

    # 預先建立新 observation 格式
    empty_obs = np.full(obs_shape, -1, dtype=all_obs.dtype)
    empty_obs[2::3] = 0
    modified_obs = np.tile(empty_obs, (num_obs, 1))

    # 取得每個代理的索引範圍
    agents_indices = [range(3 * n_fruits, 3 * n_fruits + 2) for _ in range(num_obs)]
    agents_positions = np.array([obs[indices] for obs, indices in zip(all_obs, agents_indices)])

    obj_positions = all_obs.reshape(num_obs, -1, 3)[:, :, :2]

    # 判斷是否為視野卡左/上牆的狀況
    y_stuck_dist = old_sight - agents_positions[:, 0]
    x_stuck_dist = old_sight - agents_positions[:, 1]
    y_stuck = y_stuck_dist > 0
    x_stuck = x_stuck_dist > 0

    delta_y = np.where(y_stuck, -np.minimum(y_stuck_dist - (old_sight - new_sight), 0), old_sight - new_sight)
    delta_x = np.where(x_stuck, -np.minimum(x_stuck_dist - (old_sight - new_sight), 0), old_sight - new_sight)

    new_positions_y = obj_positions[:, :, 0] - delta_y[:, np.newaxis]
    new_positions_x = obj_positions[:, :, 1] - delta_x[:, np.newaxis]
    new_positions = np.stack((new_positions_y, new_positions_x), axis=-1)

    new_agents_positions_y = agents_positions[:, 0] - delta_y
    new_agents_positions_x = agents_positions[:, 1] - delta_x

    # 計算每個代理的水果和代理範圍
    is_fruit = np.array([np.arange(obs.shape[0] // 3) < n_fruits for obs in all_obs])

    visible_fruit = is_fruit & (
            (new_agents_positions_y[:, np.newaxis] - new_sight <= new_positions[:, :, 0]) &
            (new_positions[:, :, 0] <= new_agents_positions_y[:, np.newaxis] + new_sight) &
            (new_agents_positions_x[:, np.newaxis] - new_sight <= new_positions[:, :, 1]) &
            (new_positions[:, :, 1] <= new_agents_positions_x[:, np.newaxis] + new_sight))

    visible_agent = (~is_fruit) & (0 <= new_positions[:, :, 0]) & (new_positions[:, :, 0] <= new_sight * 2) & \
                    (0 <= new_positions[:, :, 1]) & (new_positions[:, :, 1] <= new_sight * 2)

    not_visible_old = (obj_positions[:, :, 0] == -1) & (obj_positions[:, :, 1] == -1)
    visible_mask = (visible_fruit | visible_agent) & ~not_visible_old

    for i in range(num_obs):
        valid_indices = np.where(visible_mask[i])[0]

        sorted_fruit_indices = np.where(is_fruit[i][valid_indices])[0]
        sorted_agent_indices = np.where(~is_fruit[i][valid_indices][len(sorted_fruit_indices):])[0]

        fruit_indices_put = sorted_fruit_indices * 3
        agent_indices_put = sorted_agent_indices * 3 + 3 * n_fruits

        indices_to_put = np.hstack((fruit_indices_put, agent_indices_put))

        modified_obs[i][indices_to_put] = new_positions[i][valid_indices][:, 0]
        modified_obs[i][indices_to_put + 1] = new_positions[i][valid_indices][:, 1]
        modified_obs[i][indices_to_put + 2] = all_obs[i].reshape(-1, 3)[valid_indices, 2]

    return list(modified_obs)
