# This program is to check whether the rware preprocessing is correct or not.
import gym
import numpy as np
from tqdm import tqdm

from epy_tools.rware_utils import rware_preprocess, obs_parser, show_diff

# ============= What should we test? =============

# Map settings
ENV_IDS = [  # Without filling the sight
    'robotic_warehouse:rware-tiny-2ag-easy-{}s-v1',
    'robotic_warehouse:rware-tiny-4ag-easy-{}s-v1',
    'robotic_warehouse:rware-small-2ag-easy-{}s-v1',
]

# Maximum sight
MAX_SIGHT = 5

# SEEDS
SEEDS = list(range(100))


#


def get_obs_truth(sight, seed, env_id):
    np.random.seed(seed)
    env = gym.make(env_id.format(sight))
    return env.reset(), env


def get_obs_preprocessed(original_sight, sight, seed, env_id):
    np.random.seed(seed)
    env = gym.make(env_id.format(original_sight))
    obs = env.reset()
    obs = rware_preprocess(new_sight=sight, original_sight=original_sight,
                           original_obs_list=obs, verbose=False)
    return obs, env


def expand_truth_obs_size(small_sight, large_sight, seed, small_obs, large_same_shape_obs):
    """Old: smaller sight, New: larger sight
    Here, we have a small sight obs.
    We want to expand it to a larger sight obs by filling zeros around the small sight grids.
    """
    assert small_sight < large_sight, 'Original sight should be larger than the new sight'
    # Make a new one
    expanded_obs = np.zeros(large_same_shape_obs.shape)
    # Make an array which size is the same as the large sight grid info
    large_sight_grid_info = np.zeros(shape=(small_obs.shape[0], (2 * large_sight + 1) ** 2 * 7))
    # Put the first 8 elements
    expanded_obs[:, :8] = small_obs[:, :8]
    # Get small sight grid info
    small_grid_info = small_obs[:, 8: 8 + (2 * small_sight + 1) ** 2 * 7]
    # Use 2D view
    small_grid_info = small_grid_info.reshape((small_obs.shape[0], 2 * small_sight + 1, 2 * small_sight + 1, 7))
    large_grid_info = large_sight_grid_info.reshape((small_obs.shape[0], 2 * large_sight + 1, 2 * large_sight + 1, 7))
    # Put the small sight grid info to the large sight grid info
    start_idx = large_sight - small_sight
    end_idx = large_sight + small_sight + 1
    large_grid_info[:, start_idx: end_idx, start_idx: end_idx] = small_grid_info
    # Flatten it
    large_sight_grid_info = large_grid_info.reshape((small_obs.shape[0], (2 * large_sight + 1) ** 2 * 7))
    # Put the large sight grid info to the expanded obs
    expanded_obs[:, 8: 8 + (2 * large_sight + 1) ** 2 * 7] = large_sight_grid_info

    return expanded_obs


def check_correctness(original_sight, sight, seed, env_id):
    obs_truth, env_truth = get_obs_truth(sight, seed, env_id)
    obs_truth = np.array(obs_truth)
    obs_preprocessed, env_pre = get_obs_preprocessed(original_sight, sight, seed, env_id)
    obs_truth_expanded = expand_truth_obs_size(sight, original_sight, seed, obs_truth, obs_preprocessed)
    # Assert
    if not np.all(obs_truth_expanded == obs_preprocessed):
        # obs_parser(obs_truth_expanded[0], sight=original_sight)
        # obs_parser(obs_preprocessed[0], sight=original_sight)
        env_truth.render()
        env_pre.render()
        # Run this function to show the difference
        show_diff(obs_truth_expanded, obs_preprocessed, original_sight)
        obs_parser(obs_truth, sight=sight)

        raise ValueError('Preprocessing is incorrect')
    env_truth.close()
    env_pre.close()


def main():
    total_cases = sum(range(MAX_SIGHT)) * len(SEEDS) * len(ENV_IDS)
    p_bar = tqdm(total=total_cases)
    for s1 in range(2, MAX_SIGHT + 1):
        for s2 in range(1, s1):
            for seed in SEEDS:
                for env_id in ENV_IDS:
                    check_correctness(original_sight=s1, sight=s2, seed=seed, env_id=env_id)
                    p_bar.update(1)



if __name__ == "__main__":
    main()
