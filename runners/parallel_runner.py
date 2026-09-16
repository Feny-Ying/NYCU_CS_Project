from functools import partial
from multiprocessing import Pipe, Process
from typing import Optional, TYPE_CHECKING

import numpy as np
from components.episode_buffer import EpisodeBatch

from envs import REGISTRY as env_REGISTRY

if TYPE_CHECKING:
    from epy_tools.preprocess import PreprocessManager


def normalized_reward_to_go(rewards, start_t, gamma=0.99):
    """
    rewards shape: [T, n_agents]
    return shape: [n_agents]
    """
    if len(rewards) == 0 or start_t >= len(rewards):
        return np.zeros(rewards.shape[1], dtype=np.float32)

    G = np.zeros(rewards.shape[1], dtype=np.float32)
    weight_sum = 0.0
    power = 1.0

    for t in range(start_t, len(rewards)):
        G += power * rewards[t]
        weight_sum += power
        power *= gamma

    return G / max(weight_sum, 1e-8)

def segment_average_reward(rewards, start_t, end_t):
    """
    rewards shape: [T, n_agents]
    return shape: [n_agents]
    """
    if len(rewards) == 0:
        return None

    end_t = min(end_t, len(rewards))
    start_t = min(start_t, end_t)

    if start_t >= end_t:
        return None

    return np.mean(rewards[start_t:end_t], axis=0)


# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        env_args = [self.args.env_args.copy() for _ in range(self.batch_size)]
        for i in range(self.batch_size):
            env_args[i]["seed"] += i

        self.ps = [Process(target=env_worker, args=(worker_conn, CloudpickleWrapper(partial(env_fn, **env_arg))))
                   for env_arg, worker_conn in zip(env_args, self.worker_conns)]

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000

        self.preprocess_manager: Optional[PreprocessManager] = None
        self.adaptive_sight: Optional[str] = None

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        self.parent_conns[0].send(("save_replay", None))

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        # if self.args.env == 'asc2' or self.args.env == 'metadrive':
        # adaptive_sight is a str; turn it into int (e.g., 2s -> 2)

        if self.adaptive_sight is not None:
            # 假設 self.adaptive_sight 現在是 ["100m", "200m", "150m", ...]
            # 我們將每個元素都處理成整數，組成一個新的 list
            int_adaptive_sight = [int(sight[:-1]) for sight in self.adaptive_sight]
        else:
            int_adaptive_sight = None

        # Some environments' sights are controlled by this way
        if self.args.env in ['asc2', 'metadrive', 'resco2']:
            for parent_conn in self.parent_conns:
                parent_conn.send(("set_adaptive_sight", int_adaptive_sight))

        # Add sight id if needed
        if self.adaptive_sight is not None and self.args.env_args['add_sight_id_len'] is not None:
            assert self.args.env in ['gymma', 'metadrive', 'resco2', 'asc2']
            assert self.args.env not in ['sc2'], 'Not support them.'
            adaptive_sight_idx = self.preprocess_manager.get_adaptive_sight_index(adaptive_sight=self.adaptive_sight)
            for parent_conn in self.parent_conns:
                parent_conn.send(("set_cur_adaptive_sight_idx", adaptive_sight_idx))

        self.batch = self.new_batch()

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": []
        }
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["avail_actions"].append(data["avail_actions"])

            obs = data["obs"]
            obs_dict = self.preprocess_manager.get_preprocessed_obs_dict(obs, self.adaptive_sight)
            for k, v in obs_dict.items():
                if k not in pre_transition_data:
                    pre_transition_data[k] = []
                pre_transition_data[k].append(v)

            # Old: pre_transition_data["state"].append(data["state"])
            if self.args.env == 'asc2':
                if self.args.env_args['concatenate_obs_as_state']:
                    processed_state = self.preprocess_manager.get_state_by_concatenate_obs_in_obs_dict(obs_dict)
                else:
                    processed_state = data["state"]
            else:  # Other envs
                processed_state = self.preprocess_manager.get_state_by_concatenate_obs_in_obs_dict(obs_dict)

            pre_transition_data["state"].append(processed_state)

        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False):
        # Adaptive sight settings
        cur_time_bin, self.adaptive_sight = self.preprocess_manager.try_get_temporal_adaptive_sights(
            greedy=test_mode,
            t_env=self.t_env,
            t_ep=0,
            episode_limit=self.episode_limit,
        )
        if self.adaptive_sight is not None:
            # 假設 self.adaptive_sight 現在是 ["100m", "200m", "150m", ...]
            # 我們將每個元素都處理成整數，組成一個新的 list
            int_adaptive_sight = [int(sight[:-1]) for sight in self.adaptive_sight]
        else:
            int_adaptive_sight = None
        self.reset()

        self.parent_conns[0].send(("record_gif", None))

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]

        agent_episode_returns = np.zeros((self.batch_size, self.env_info["n_agents"]))

        agent_reward_trajs = [[] for _ in range(self.batch_size)]

        sight_records = [[] for _ in range(self.batch_size)]

        for b in range(self.batch_size):
            sight_records[b].append({
                "start_t": 0,
                "time_bin": cur_time_bin,
                "sights": list(self.adaptive_sight),
            })

        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        while True:
            obs_key = self.preprocess_manager.get_obs_key(t_env=self.t_env)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated,
                                              test_mode=test_mode, obs_key=obs_key)
            cpu_actions = actions.to("cpu").numpy()

            # Update the actions taken
            actions_chosen = {
                "actions": actions.unsqueeze(1)
            }
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # 下一個 obs 對應的是 t + 1
            next_t = self.t + 1

            new_time_bin = self.preprocess_manager.get_temporal_ucb_bin(
                t_ep=next_t,
                episode_limit=self.episode_limit,
            )

            # 進入新的 time bin 才重新選 sight
            if self.adaptive_sight is not None and new_time_bin != cur_time_bin:
                cur_time_bin, self.adaptive_sight = self.preprocess_manager.try_get_temporal_adaptive_sights(
                    greedy=test_mode,
                    t_env=self.t_env,
                    t_ep=next_t,
                    episode_limit=self.episode_limit,
                )

                for b in envs_not_terminated:
                    sight_records[b].append({
                        "start_t": next_t,
                        "time_bin": cur_time_bin,
                        "sights": list(self.adaptive_sight),
                    })

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated:  # We produced actions for this env
                    if not terminated[idx]:  # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1  # actions is not a list over every env
                    # if self.args.render:
                    #     parent_conn.send(("render", None))
                    if self.args.force_render is True and idx == 0:
                        parent_conn.send(("render", None))
                    else:
                        if idx == 0 and test_mode and self.args.render:
                            parent_conn.send(("render", None))

                    # if self.args.force_render is True:
                    #     self.env.render()
                    # else:
                    #     if test_mode and self.args.render:
                    #         self.env.render()

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "terminated": []
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": []
            }

            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["reward"],))

                    episode_returns[idx] += data["reward"]

                    if "local_rewards" in data["info"]:
                        local_rewards = np.array(data["info"]["local_rewards"], dtype=np.float32)

                        # 原本整局總和保留
                        agent_episode_returns[idx] += local_rewards

                        # 新增：每一步 local reward 都存起來，episode end 後算 reward-to-go
                        agent_reward_trajs[idx].append(local_rewards)
                        
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                        # Only for MetaDrive
                        if idx == 0:
                            if test_mode and self.args.render and not self.args.force_render:
                                parent_conn.send(("save_metadrive_gif", None))

                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["avail_actions"].append(data["avail_actions"])

                    obs = data["obs"]
                    obs_dict = self.preprocess_manager.get_preprocessed_obs_dict(obs, self.adaptive_sight)
                    for k, v in obs_dict.items():
                        if k not in pre_transition_data:
                            pre_transition_data[k] = []
                        pre_transition_data[k].append(v)

                    # Old: pre_transition_data["state"].append(data["state"])
                    if self.args.env == 'asc2':
                        if self.args.env_args['concatenate_obs_as_state']:
                            processed_state = self.preprocess_manager.get_state_by_concatenate_obs_in_obs_dict(obs_dict)
                        else:
                            processed_state = data["state"]
                    else:  # Other envs
                        processed_state = self.preprocess_manager.get_state_by_concatenate_obs_in_obs_dict(obs_dict)
                    pre_transition_data["state"].append(processed_state)

                    # # Print original obs, state, and obs_dict['obs'], processed_state
                    # print(f'obs:\n{obs}')
                    # print(f'state:\n{data["state"]}')
                    # print(f'obs_dict[obs]:\n{obs_dict["obs"]}')
                    # print(f'processed_state:\n{processed_state}')
                    # raise Exception('stop here')

            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Update the adaptive worker using training episode returns
        # TODO: whether use only training episodes?
        # segment average reward for each sight segment
        if self.adaptive_sight is not None:
            if not test_mode:
                for b in range(self.batch_size):
                    rewards = np.array(agent_reward_trajs[b], dtype=np.float32)

                    if rewards.size == 0:
                        continue

                    for i, record in enumerate(sight_records[b]):
                        start_t = record["start_t"]
                        time_bin = record["time_bin"]
                        sights = record["sights"]

                        # 下一個 sight record 的 start_t 就是這段 segment 的結束
                        if i + 1 < len(sight_records[b]):
                            end_t = sight_records[b][i + 1]["start_t"]
                        else:
                            end_t = len(rewards)

                        segment_reward = segment_average_reward(
                            rewards=rewards,
                            start_t=start_t,
                            end_t=end_t,
                        )

                        if segment_reward is None:
                            continue

                        self.preprocess_manager.update_temporal_adaptive_agents(
                            time_bin=time_bin,
                            arm_names=sights,
                            rewards=segment_reward,
                        )
        # Update the adaptive worker using training episode returns
        # if self.adaptive_sight is not None:
        #     if not test_mode:
        #         gamma = getattr(self.args, "temporal_ucb_gamma", 0.99)

        #         for b in range(self.batch_size):
        #             rewards = np.array(agent_reward_trajs[b], dtype=np.float32)

        #             if rewards.size == 0:
        #                 continue

        #             for record in sight_records[b]:
        #                 start_t = record["start_t"]
        #                 time_bin = record["time_bin"]
        #                 sights = record["sights"]

        #                 rtg = normalized_reward_to_go(
        #                     rewards=rewards,
        #                     start_t=start_t,
        #                     gamma=gamma,
        #                 )

        #                 self.preprocess_manager.update_temporal_adaptive_agents(
        #                     time_bin=time_bin,
        #                     arm_names=sights,
        #                     rewards=rtg,
        #                 )

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats", None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)
        # print(f'env_stats: {env_stats}')
        # print(f'final_env_infos: {final_env_infos}')

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        # infos = [cur_stats] + final_env_infos

        # print('----------------------------------------~~~~~')
        # print(f'[cur_stats] {[cur_stats]}')
        # print(f'final_env_infos {final_env_infos}')
        # print(f'env_stats {env_stats}')

        infos = [cur_stats] + final_env_infos + env_stats
        
        # 建立一個新的統計字典，過濾掉無法直接加總的非數值資料
        new_stats = {}
        all_keys = set.union(*[set(d) for d in infos])
        
        for k in all_keys:
            # 找到第一個有值的資料來判斷型別
            first_val = next((d[k] for d in infos if k in d), None)
            
            # 只有當資料是數字 (int, float, np.number) 時才進行 sum
            if isinstance(first_val, (int, float, np.number)):
                new_stats[k] = sum(d.get(k, 0) for d in infos)
            else:
                # 如果是 list (如 local_rewards) 或其他非數值，不進行全局統計加總
                continue
        
        cur_stats.update(new_stats)
        
        # 以下原本的邏輯維持不變
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        if test_mode:
            if int_adaptive_sight is not None:
                # int_adaptive_sight 是 [100, 50, 150...]
                for i, sight in enumerate(int_adaptive_sight):
                    # 替每個 agent 建立獨立的 key
                    key = f'selected_sight_agent_{i}'
                    # 累加該 agent 的距離 (乘以 batch_size 是為了之後算平均)
                    cur_stats[key] = (sight * self.batch_size) + cur_stats.get(key, 0)
                
                # 同時保留一個總平均值，方便看整體趨勢
                avg_sight = sum(int_adaptive_sight) / len(int_adaptive_sight)
                cur_stats['selected_sight'] = (avg_sight * self.batch_size) + cur_stats.get('selected_sight', 0)

        cur_returns.extend(episode_returns)

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
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


def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            reward, terminated, env_info = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                # Rest of the data for the current timestep
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            env.reset()
            remote.send({
                "state": env.get_state(),
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs()
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        elif cmd == "render":
            # if 'MetaDriveMA' in env.key:
            #     env.original_env.render(mode="top_down",
            #                             scaling=4,  # 4 pixels per meter
            #                             screen_size=(500, 500))
            # else:
            env.render()
        elif cmd == "save_metadrive_gif":
            if 'MetaDriveMA' in env.key:
                env.original_env.save_metadrive_gif()
        elif cmd == "record_gif":
            # if env has a "key" attribute, check it
            if hasattr(env, "key"):
                if 'MetaDriveMA' in env.key:
                    env.original_env.set_record_gif()
        elif cmd == "save_replay":
            env.save_replay()
        elif cmd == "set_cur_adaptive_sight_idx":
            # print(f'set_cur_adaptive_sight_idx: {data}')
            env.set_cur_adaptive_sight_idx(data)
        elif cmd == "set_adaptive_sight":
            # print(f'set_adaptive_sight: {data}')
            env.set_adaptive_sight(data)
        else:
            raise NotImplementedError


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)
