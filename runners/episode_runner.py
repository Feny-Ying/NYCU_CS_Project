from functools import partial
from typing import Optional, TYPE_CHECKING

import numpy as np
from components.episode_buffer import EpisodeBatch

from envs import REGISTRY as env_REGISTRY

if TYPE_CHECKING:
    from epy_tools.preprocess import PreprocessManager


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

        self.preprocess_manager: Optional[PreprocessManager] = None

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        adaptive_sight: Optional[str] = self.preprocess_manager.try_get_adaptive_sight(greedy=test_mode)
        # adaptive_sight is a str; turn it into int (e.g., 2s -> 2)

        if adaptive_sight is not None:
            int_adaptive_sight = int(adaptive_sight[:-1])
        else:
            int_adaptive_sight = None

        # Some environments' sights are controlled by this way
        if self.args.env in ['asc2', 'metadrive', 'resco2']:
            self.env.set_adaptive_sight(int_adaptive_sight)

        # Some environments support adding sight id one-hot
        if adaptive_sight is not None and self.args.env_args['add_sight_id_len'] is not None:
            assert self.args.env in ['gymma', 'metadrive', 'resco2', 'asc2']
            assert self.args.env not in ['sc2'], 'Not support them.'
            adaptive_sight_idx = self.preprocess_manager.get_adaptive_sight_index(adaptive_sight=adaptive_sight)
            # print(f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ setting adaptive_sight_idx: {adaptive_sight_idx}')
            self.env.set_cur_adaptive_sight_idx(adaptive_sight_idx)


        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            # batch_for_select_action = self.preprocess_manager.get_batch_for_select_action_and_update_cur_batch_for_save(
            #     state=self.env.get_state(),
            #     avail_actions=self.env.get_avail_actions(),
            #     original_obs=self.env.get_obs(),
            #     t_env=self.t_env,
            #     t=self.t,
            #     cur_batch=self.batch,
            # )

            pre_data = {"avail_actions": [self.env.get_avail_actions()]}

            # Preprocess obs
            obs_dict = self.preprocess_manager.get_preprocessed_obs_dict(self.env.get_obs(), adaptive_sight)
            obs_key = self.preprocess_manager.get_obs_key(t_env=self.t_env)
            pre_data.update({k: [v] for k, v in obs_dict.items()})

            # Preprocess state
            if self.args.env == 'asc2':
                if self.args.env_args['concatenate_obs_as_state']:
                    # TODO: for the state part of POEM, the ParallelRunner hasn't been modify yet
                    processed_state = self.preprocess_manager.get_state_by_concatenate_obs_in_obs_dict(obs_dict)
                else:
                    processed_state = self.env.get_state()
            else:  # Other envs
                processed_state = self.preprocess_manager.get_state_by_concatenate_obs_in_obs_dict(obs_dict, obs_key)
            pre_data.update({"state": [processed_state]})

            # TODO: implement on ParallelRunner, too

            self.batch.update(pre_data, ts=self.t)

            # batch_for_select_action = self.preprocess_manager.get_batch_for_select_action_and_update_cur_batch_for_save(
            #     state=self.env.get_state(),
            #     avail_actions=self.env.get_avail_actions(),
            #     original_obs=self.env.get_obs(),
            #     t_env=self.t_env,
            #     t=self.t,
            #     cur_batch=self.batch,
            # )

            # 將 ``batch_for_select_action['obs'][0] 存到檔案 "jjj.txt" 中
            # 若存在，則append (一行一行)
            # print(f'obs_dict["obs"][0].shape: {obs_dict["obs"][0].shape}')

            # with open("jjj.txt", "a") as f:
            #     # f.write(f"{obs_dict['obs'][0].cpu().detach().numpy().tolist()}\n")
            #     f.write(f"{obs_dict['obs'][0].tolist()}\n")

            # with open("jjj2.txt", "a") as f:
            #     f.write(f"{obs_dict['obs_1s'][0].tolist()}\n")

            # with open("jjj2s.txt", "a") as f:
            #     f.write(f"{batch_for_select_action['obs_1s'][0][0][0].cpu().detach().numpy().tolist()}\n")
            # # 若檔案 size 超過 1000 行，則raise
            # with open("jjj.txt", "r") as f:
            #     if len(f.readlines()) > 10000:
            #         raise ValueError("Too many lines in file 'jjj.txt'")
            # with open("jjj2s.txt", "r") as f:
            #     if len(f.readlines()) > 10000:
            #         raise ValueError("Too many lines in file 'jjj.txt'")

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            # actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            # actions = self.mac.select_actions(batch_for_select_action, t_ep=self.t, t_env=self.t_env,
            #                                   test_mode=test_mode)
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env,
                                              test_mode=test_mode, obs_key=obs_key)
            reward, terminated, env_info = self.env.step(actions[0])
            if self.args.force_render is True:
                self.env.render()
            else:
                if test_mode and self.args.render:
                    self.env.render()
            episode_return += reward

            pre_transition_data_for_buffer = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(pre_transition_data_for_buffer, ts=self.t)

            self.t += 1

        #
        # batch_for_select_action = self.preprocess_manager.get_batch_for_select_action_and_update_cur_batch_for_save(
        #     state=self.env.get_state(),
        #     avail_actions=self.env.get_avail_actions(),
        #     original_obs=self.env.get_obs(),
        #     t_env=self.t_env,
        #     t=self.t,
        #     cur_batch=self.batch,
        # )

        pre_data = {"avail_actions": [self.env.get_avail_actions()]}

        obs_dict = self.preprocess_manager.get_preprocessed_obs_dict(self.env.get_obs(), adaptive_sight)
        obs_key = self.preprocess_manager.get_obs_key(t_env=self.t_env)
        pre_data.update({k: [v] for k, v in obs_dict.items()})

        # Preprocess state
        if self.args.env == 'asc2':
            if self.args.env_args['concatenate_obs_as_state']:
                processed_state = self.preprocess_manager.get_state_by_concatenate_obs_in_obs_dict(obs_dict)
            else:
                processed_state = self.env.get_state()
        else:  # Other envs
            processed_state = self.preprocess_manager.get_state_by_concatenate_obs_in_obs_dict(obs_dict, obs_key)
        pre_data.update({"state": [processed_state]})

        self.batch.update(pre_data, ts=self.t)

        if test_mode and self.args.render:
            print(f"Episode return: {episode_return}")

        # Select actions in the last stored state
        # actions = self.mac.select_actions(batch_for_select_action, t_ep=self.t, t_env=self.t_env,
        #                                   test_mode=test_mode)
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env,
                                          test_mode=test_mode, obs_key=obs_key)
        self.batch.update({"actions": actions}, ts=self.t)

        # TODO: note that "get_stats" is not implemented in this runner!!!!!!!!!!! 🔥
        # print(f'env_info: {env_info}')

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if test_mode:
            # cur_stats['selected_sight'] = int_adaptive_sight
            if int_adaptive_sight is not None:
                cur_stats['selected_sight'] = int_adaptive_sight + cur_stats.get('selected_sight', 0)


        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        # Update UCB
        if adaptive_sight is not None:
            # self.preprocess_manager.adaptive_worker.update(arm_name=adaptive_sight, reward=episode_return)
            if not test_mode:
                self.preprocess_manager.adaptive_worker.update(arm_name=adaptive_sight, reward=episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
        stats.clear()
