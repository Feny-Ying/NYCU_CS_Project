import copy
from typing import Optional, TYPE_CHECKING

import numpy as np
import torch
from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from modules.mixers.qmix import QMixer
from modules.mixers.vdn import VDNMixer
from torch.optim import Adam

if TYPE_CHECKING:
    from epy_tools.preprocess import PreprocessManager


def reset_weights(m):
    """
    這個函數將根據層的類型重新初始化權重
    """
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()


import torch as th


def farthest_agent_selection(norm_agent_obs, n_picked):
    """
    FAS 演算法選擇距離最遠的 lambda 個 agents，基於歐幾里得距離。

    參數:
    - norm_agent_obs: [n_agents, obs_size] 張量，每個 agent 的平均觀測值。
    - n_picked: 需要選擇的 agent 數量 lambda。

    返回:
    - selected_agents: 被選中的 agent 的索引列表。
    """
    n_agents = norm_agent_obs.shape[0]  # agents 的數量
    selected_agents = []

    # 1. 初始化，隨機選擇第一個 agent 作為起點
    first_agent = th.randint(0, n_agents, (1,)).item()
    selected_agents.append(first_agent)

    # 2. 迭代選擇其他的 agents
    for _ in range(n_picked - 1):
        max_dist = -float('inf')
        next_agent = None

        # 迭代所有尚未選中的 agents，選擇與當前已選 agents 距離最遠的
        for agent_id in range(n_agents):
            if agent_id in selected_agents:
                continue

            # 計算該 agent 與所有已選 agents 的最小距離（因為我們想要挑選最遠的）
            min_dist = float('inf')
            for selected_agent in selected_agents:
                # 使用歐幾里得距離 (L2 norm) 計算兩個觀測之間的距離
                dist = th.norm(norm_agent_obs[agent_id] - norm_agent_obs[selected_agent], p=2).item()
                min_dist = min(min_dist, dist)

            # 更新與當前已選 agents 距離最遠的 agent
            if min_dist > max_dist:
                max_dist = min_dist
                next_agent = agent_id

        selected_agents.append(next_agent)

    return selected_agents


class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())
        self.last_target_update_episode = 0

        self.mixer = None
        # assert args.mixer is None, "No mixer supported currently."
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = Adam(params=self.params, lr=args.lr)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.training_steps = 0
        self.last_target_update_step = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=device)

        # Teacher (last itself)
        self.teacher_mac_agent = None
        self.teacher_mixer = None

        # Policy diff
        self.prev1_mac_agent_for_policy_diff = None
        self.prev2_mac_agent_for_policy_diff = None
        self.prev3_mac_agent_for_policy_diff = None
        self.prev1_epsilon = None
        self.prev2_epsilon = None
        self.prev3_epsilon = None
        self.tv_list_1 = []
        self.tv_list_2 = []

        if self.args.policy_diff_interval_2:
            assert self.args.policy_diff_interval_1, "Policy diff interval 1 must be enabled."
            assert self.args.policy_diff_interval_2 > self.args.policy_diff_interval_1, "Policy diff interval 2 must be larger than interval 1."
        if self.args.policy_diff_interval_3:
            assert self.args.policy_diff_interval_2, "Policy diff interval 2 must be enabled."
            assert self.args.policy_diff_interval_3 > self.args.policy_diff_interval_2, "Policy diff interval 3 must be larger than interval 2."

        self.preprocess_manager: Optional[PreprocessManager] = None

        # For FAS Dynamically adjust lambda
        self.cumulative_avg_variance = None
        self.fas_i_episode = 0

    def train(self, batch: EpisodeBatch, t_env: int, assigned_obs_key: Optional[str] = None):
        # Keep the original mac and mixer
        if self.args.policy_diff_interval_1 and self.prev1_mac_agent_for_policy_diff is None:
            self.prev1_mac_agent_for_policy_diff = copy.deepcopy(self.mac.agent)
            self.prev1_epsilon = self.args.epsilon_start
        if self.args.policy_diff_interval_2 and self.prev2_mac_agent_for_policy_diff is None:
            self.prev2_mac_agent_for_policy_diff = copy.deepcopy(self.mac.agent)
            self.prev2_epsilon = self.args.epsilon_start
        if self.args.policy_diff_interval_3 and self.prev3_mac_agent_for_policy_diff is None:
            self.prev3_mac_agent_for_policy_diff = copy.deepcopy(self.mac.agent)
            self.prev3_epsilon = self.args.epsilon_start

        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # --------------- FAS Post-Processing ----------------------------

        agent_selected_count, all_avg_variance, all_lambda_value, lambda_value, variance_ratio = None, None, None, None, None
        if self.args.use_fas:
            if not self.args.env in ['asc2']:
                raise ValueError("FAS is only supported for ASC2 environment.")
            if not self.args.env_args['obs_instead_of_state']:
                raise ValueError("FAS is only supported for obs instead of state.")

            # 1. 計算每個 episode 中每個 agent 的有效觀測值(用整場的steps的obs平均)並normalize
            # 2. 之後用 FAS 演算法選擇距離最遠的 lambda 個 agents
            batch_steps_agent_obs = batch['obs']
            selected_agent_ids = []
            all_avg_variance = []
            all_norm_obs = []
            for i_episode in range(batch.batch_size):
                # print(f'============ i_episode={i_episode} ============')
                steps_agent_obs = batch_steps_agent_obs[i_episode]  # [steps, agent, obs]
                # 假設有 m 個 steps，mask[i] 為形狀 m-1 的 tensor，其中 1 表示有效，0 表示無效
                # 用 1 的數量來計算有效步驟數，然後取出 obs
                n_valid_steps = int(mask[i_episode, ...].sum().item())
                # print(f'n_valid_steps: {n_valid_steps}')
                valid_obs = steps_agent_obs[:n_valid_steps, ...]  # [n_valid_steps, agent, obs]
                # print(f'valid_obs.shape: {valid_obs.shape}')
                avg_obs = valid_obs.mean(dim=0)  # 在有效步上取平均，結果形狀 [agent, obs]
                # print(f'avg_obs.shape: {avg_obs.shape}')
                # Element-wise Normalize。也就是obs的第k個元素都用所有agent的第k個元素的最大最小值來normalize
                normalized_avg_obs = th.zeros_like(avg_obs)  # [n_agents, obs_size]
                for k in range(avg_obs.shape[1]):
                    max_val = avg_obs[:, k].max()
                    min_val = avg_obs[:, k].min()
                    # 處理 max_val == min_val 的情況
                    if max_val == min_val:
                        normalized_avg_obs[:, k] = 0.0  # 如果 max == min，則該維度上的所有元素都設為 0
                    else:
                        normalized_avg_obs[:, k] = (avg_obs[:, k] - min_val) / (max_val - min_val + 1e-8)
                all_norm_obs.append(normalized_avg_obs)
                # 笡每個 element-wise 的 variance, 之後算平均variance
                normalized_avg_obs_var = normalized_avg_obs.var(dim=0)
                avg_obs_var = normalized_avg_obs_var.mean().item()
                all_avg_variance.append(avg_obs_var)

            # 動態調整 λ (第一次先用固定的 λ)
            this_batch_avg_variance = np.mean(all_avg_variance)
            if self.cumulative_avg_variance is None:
                variance_ratio = None
            else:
                variance_ratio = this_batch_avg_variance / (self.cumulative_avg_variance + 1e-8)
            if self.cumulative_avg_variance and self.args.fas_dynamic:
                # 根據當前 episode 的觀測方差與累積平均方差的比例來動態調整 λ
                lambda_value = int(min(self.args.fas_lambda, max(1, self.args.n_agents * variance_ratio)))
            else:
                lambda_value = int(self.args.fas_lambda)

            # FAS 演算法選擇距離最遠的 lambda 個 agents
            for normalized_avg_obs in all_norm_obs:
                selected_agent_ids += farthest_agent_selection(normalized_avg_obs, n_picked=lambda_value)

            # Update cumulative variance
            if self.cumulative_avg_variance is None:
                self.cumulative_avg_variance = this_batch_avg_variance
            else:
                self.cumulative_avg_variance = (
                                                       self.cumulative_avg_variance * self.fas_i_episode + this_batch_avg_variance) / (
                                                       self.fas_i_episode + 1)
            self.fas_i_episode += 1

            # 3. 統計被選次數最多的 top-lambda 個 agents
            n_agent = batch['obs'].shape[2]
            agent_selected_count = th.zeros(n_agent, dtype=th.int32)  # 初始化一個計數張量
            for selected_agents in selected_agent_ids:
                agent_selected_count[selected_agents] += 1  # 將每個 episode 中被選中的 agent 次數加1

            # 取出被選次數最多的 lambda 個 agents
            top_agents = th.argsort(agent_selected_count, descending=True)[:lambda_value]  # 被選次數最多的前 lambda 個 agents
            # print(f'Each agent selected count: {agent_selected_count}')
            # print(f'Top agents selected: {top_agents}')

            # 4. 用 agent obs concat 起來做成 state，但覆蓋掉沒被選到的部分用 -1
            state_per_episode = []  # 用來儲存每個 episode 的 state
            for i_episode in range(batch.batch_size):
                episode_state = []  # 儲存每個 episode 的 state
                for agent_id in range(self.args.n_agents):
                    agent_obs = batch["obs"][i_episode, :, agent_id, :]  # 提取這個 agent 的所有時間步觀測數據
                    if agent_id not in top_agents:
                        # 如果 agent 沒被選中，覆蓋它的觀測數據為 -1
                        agent_obs = th.full_like(agent_obs, -1)

                    episode_state.append(agent_obs)  # 將 agent 的觀測數據加入到 episode 的 state 中

                # 將這個 episode 中所有 agent 的觀測數據進行 concat，形成該 episode 的 state
                episode_state = th.cat(episode_state, dim=-1)  # 最後一個維度做 concat, 結果形狀是 [steps, obs_concat]
                state_per_episode.append(episode_state)

            # 最終將所有 episodes 的 state 合併成一個張量
            final_state = th.stack(state_per_episode)  # [batch_size, steps, state_size]

            # # 印出第一個episode的第一步
            # np.set_printoptions(threshold=np.inf, linewidth=np.inf,)
            # print(f'final_state[0][0]: {final_state[0][0]}')

            # 5. 將 state 覆蓋到 batch 中
            # batch['state'] = final_state
            # print(f'Before [0][0]: {batch["state"].cpu().numpy()[0][0]}')
            batch.update({'state': final_state})
            # print(f'After  [0][0]: {batch["state"].cpu().numpy()[0][0]}')

            # print(batch['state'])
            # print(f'filled: {batch["filled"]}')
            # print(f'filled.shape: {batch["filled"].shape}')
            # print(f'terminal.shape: {batch["terminated"].shape}')
            #
            # print(f'filled.sum(axis=1): {batch["filled"].sum(axis=1)}')
            # print(f'mask.sum(axis=1): {mask.sum(axis=1)}')
            # print(f'terminal: {(1-batch["terminated"]).sum(axis=1)}')
            # print(f'terminal: {batch["terminated"]}')
            # print(batch['state'].shape)
            # raise ValueError("Stop here.")

        # 前面 mask 為什麼要這樣處理，因為我們只需要計算到終止前一步的 Q 值，所以後面的步驟不需要計算
        # 仔細介紹：
        # 1. mask[:, 1:] 這樣處理是為了將終止步的 mask 設為 0，這樣在計算 Q 值時，終止步的 Q 值就不會被計算
        # 2. mask[:, 1:] * (1 - terminated[:, :-1]) 這樣處理是為了將終止步的 mask 設為 0，這樣在計算 Q 值時，終止步的 Q 值就不會被計算
        # 為什麼要有 2 呢？因為有些終止步可能是因為達到最大步數而終止，這樣的終止步我們是需要計算 Q 值的

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        # Obs key for this batch
        if assigned_obs_key is not None:
            cur_obs_key = assigned_obs_key
        else:
            cur_obs_key = self.preprocess_manager.get_obs_key(t_env)

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t, dormant_record=None, obs_key=cur_obs_key)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t, dormant_record=None, obs_key=cur_obs_key)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        if self.args.standardise_returns:
            target_max_qvals = target_max_qvals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals.detach()

        if self.args.standardise_returns:
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask
        td_loss = (masked_td_error ** 2).sum() / mask.sum()

        # Weight Decay (WD) loss
        wd_coef = float(self.args.weight_decay) if self.args.weight_decay is not None else 0
        # print(f'wd_coef: {wd_coef}')
        wd_loss = 0
        # Use MSE to calculate the avg WD loss
        mse_crit = th.nn.MSELoss()
        n_params = 0
        for param in self.mac.parameters():
            wd_loss += mse_crit(param, th.zeros_like(param))
            n_params += 1
        wd_loss /= n_params

        # Normal L2 loss, take mean over actual data
        loss = td_loss + wd_loss * wd_coef

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.training_steps += 1
        if self.args.target_update_interval_or_tau > 1 and (
                self.training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.last_target_update_step = self.training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("td_loss", td_loss.item(), t_env)
            self.logger.log_stat("wd_loss", wd_loss.item(), t_env)
            self.logger.log_stat("wd_coef", wd_coef, t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            if self.args.use_fas:
                sum_selected_count = agent_selected_count.sum().item()
                for i_agent in range(self.args.n_agents):
                    self.logger.log_stat(f"FAS/agent_{i_agent}_selected_ratio",
                                         agent_selected_count[i_agent].item() / sum_selected_count, t_env)
                self.logger.log_stat("FAS/avg_variance", np.mean(all_avg_variance), t_env)
                self.logger.log_stat("FAS/lambda", lambda_value, t_env)
                self.logger.log_stat("FAS/cumulative_avg_variance", self.cumulative_avg_variance, t_env)
                if variance_ratio:
                    self.logger.log_stat("FAS/variance_ratio", variance_ratio, t_env)

            # self.log_stats_t = t_env

    def distillation_train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        assert self.teacher_mac_agent is not None, "Teacher MAC is not set."
        if self.mixer is not None:
            assert self.teacher_mixer is not None, "Teacher mixer is not set."
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        # Calculate estimated Q-Values
        cur_obs_key = self.preprocess_manager.get_obs_key(t_env)
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t, dormant_record=None, obs_key=cur_obs_key)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Store current agent
        backup_agent = self.mac.agent

        # Calculated teacher's (computing target, without gradient)
        self.mac.agent = self.teacher_mac_agent
        teacher_mac_out = []
        with th.no_grad():
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                teacher_agent_outs = self.mac.forward(batch, t=t, dormant_record=None, obs_key=cur_obs_key)
                teacher_mac_out.append(teacher_agent_outs)
        teacher_mac_out = th.stack(teacher_mac_out, dim=1)  # Concat over time
        # Pick the Q-Values for the actions taken by each agent
        teacher_chosen_action_qvals = th.gather(teacher_mac_out[:, :-1], dim=3, index=actions).squeeze(
            3)  # Remove the last dim

        # Restore current agent
        self.mac.agent = backup_agent

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            teacher_chosen_action_qvals = self.teacher_mixer(teacher_chosen_action_qvals, batch["state"][:, :-1])

        # Td-error
        td_error = (chosen_action_qvals - teacher_chosen_action_qvals.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask
        td_loss = (masked_td_error ** 2).sum() / mask.sum()

        # Weight Decay (WD) loss
        wd_coef = float(self.args.weight_decay) if self.args.weight_decay is not None else 0
        # print(f'wd_coef: {wd_coef}')
        wd_loss = 0
        # Use MSE to calculate the avg WD loss
        mse_crit = th.nn.MSELoss()
        n_params = 0
        for param in self.mac.parameters():
            wd_loss += mse_crit(param, th.zeros_like(param))
            n_params += 1
        wd_loss /= n_params

        # Normal L2 loss, take mean over actual data
        loss = td_loss + wd_loss * wd_coef

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.training_steps += 1
        if self.args.target_update_interval_or_tau > 1 and (
                self.training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.last_target_update_step = self.training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("distillation_train/loss", loss.item(), t_env)
            self.logger.log_stat("distillation_train/td_loss", td_loss.item(), t_env)
            self.logger.log_stat("distillation_train/wd_loss", wd_loss.item(), t_env)
            self.logger.log_stat("distillation_train/wd_coef", wd_coef, t_env)
            self.logger.log_stat("distillation_train/grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("distillation_train/td_error_abs", (masked_td_error.abs().sum().item() / mask_elems),
                                 t_env)
            self.logger.log_stat("distillation_train/q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("distillation_train/target_mean",
                                 (teacher_chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            # self.log_stats_t = t_env

    def offline_train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        # Calculate estimated Q-Values
        cur_obs_key = self.preprocess_manager.get_obs_key(t_env)
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t, dormant_record=None, obs_key=cur_obs_key)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t, dormant_record=None, obs_key=cur_obs_key)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        if self.args.standardise_returns:
            target_max_qvals = target_max_qvals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals.detach()

        if self.args.standardise_returns:
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask
        td_loss = (masked_td_error ** 2).sum() / mask.sum()

        # Weight Decay (WD) loss
        wd_coef = float(self.args.weight_decay) if self.args.weight_decay is not None else 0
        # print(f'wd_coef: {wd_coef}')
        wd_loss = 0
        # Use MSE to calculate the avg WD loss
        mse_crit = th.nn.MSELoss()
        n_params = 0
        for param in self.mac.parameters():
            wd_loss += mse_crit(param, th.zeros_like(param))
            n_params += 1
        wd_loss /= n_params

        # Normal L2 loss, take mean over actual data
        loss = td_loss + wd_loss * wd_coef

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.training_steps += 1
        if self.args.target_update_interval_or_tau > 1 and (
                self.training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.last_target_update_step = self.training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("offline_train/loss", loss.item(), t_env)
            self.logger.log_stat("offline_train/td_loss", td_loss.item(), t_env)
            self.logger.log_stat("offline_train/wd_loss", wd_loss.item(), t_env)
            self.logger.log_stat("offline_train/wd_coef", wd_coef, t_env)
            self.logger.log_stat("offline_train/grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("offline_train/td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("offline_train/q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("offline_train/target_mean",
                                 (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            # self.log_stats_t = t_env

    def check_policy_diff(self, buffer, t_env: int, episode_num: int, interval_id: int):
        if interval_id == 1:
            prev_epsilon = self.prev1_epsilon
            prev_agent = self.prev1_mac_agent_for_policy_diff
        elif interval_id == 2:
            prev_epsilon = self.prev2_epsilon
            prev_agent = self.prev2_mac_agent_for_policy_diff
        elif interval_id == 3:
            prev_epsilon = self.prev3_epsilon
            prev_agent = self.prev3_mac_agent_for_policy_diff
        else:
            raise ValueError("Invalid interval_id.")
        assert prev_agent is not None, "Previous agent is not set."

        cur_agent = self.mac.agent
        cur_epsilon = self.mac.action_selector.schedule.eval(t_env)

        kl_divs = []
        tv_divs = []
        entropies = []
        q_diffs = []

        for i_batch in range(self.args.policy_diff_n_batches):
            episode_sample = self.preprocess_manager.sample_current_needed_data(buffer, self.args, t_env)
            # episode_sample = buffer.sample(self.args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != self.args.device:
                episode_sample.to(self.args.device)

            # ----------------------------------------------
            batch = episode_sample

            # Get the relevant quantities
            terminated = batch["terminated"][:, :-1].float()
            mask = batch["filled"][:, :-1].float()
            mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
            avail_actions = batch["avail_actions"]

            with (torch.no_grad()):

                # Epsilon
                cur_probs, cur_unmasked_q = self.compute_epsilon_greedy_policy(batch, cur_agent, avail_actions,
                                                                               cur_epsilon, t_env)
                prev_probs, prev_unmasked_q = self.compute_epsilon_greedy_policy(batch, prev_agent, avail_actions,
                                                                                 prev_epsilon, t_env)

                # Calculate KL divergence (use log to compute)
                prev_log_probs = th.log(prev_probs)
                cur_log_probs = th.log(cur_probs)
                kl_div = (prev_log_probs - cur_log_probs) * prev_probs
                kl_div = kl_div.sum(dim=3)  # shape: [batch_size, seq_len, n_agents]
                kl_divs.append(kl_div.mean().item())

                # Calculate TV divergence
                tv_div = 0.5 * (cur_probs - prev_probs).abs().sum(dim=3)
                tv_divs.append(tv_div.mean().item())

                # Compute current entropy
                cur_entropy = -(cur_probs * cur_log_probs).sum(dim=3)
                entropies.append(cur_entropy.mean().item())

                # Compute Q-Value difference
                q_diff = (cur_unmasked_q - prev_unmasked_q).abs()
                # Filter out unavailable actions
                q_diff[avail_actions == 0.0] = 0.0
                # Average over available actions
                # print(f'avail_actions.shape', avail_actions.shape)
                # print(q_diff.max())
                # print(q_diff.min())
                q_diff = q_diff.sum(dim=3) / avail_actions.sum(dim=3)  # Assume at least one action is available
                # print('--', q_diff.max())
                # print('--', q_diff.min())
                # print(avail_actions[:,-1,:,:].sum(axis=[-1,-2]))
                # Remove those steps that have no available actions
                idx_available_state = avail_actions.sum(dim=3) > 0
                q_diff = q_diff[idx_available_state]
                q_diffs.append(q_diff.mean().item())

        self.logger.log_stat(f"policy_diff/KL_{interval_id}", np.mean(kl_divs), t_env)
        self.logger.log_stat(f"policy_diff/n_batches_{interval_id}", self.args.policy_diff_n_batches, t_env)
        self.logger.log_stat(f"policy_diff/entropy_{interval_id}", np.mean(entropies), t_env)
        self.logger.log_stat(f"policy_diff/TV_{interval_id}", np.mean(tv_divs), t_env)
        self.logger.log_stat(f"policy_diff/Q_diff_{interval_id}", np.mean(q_diffs), t_env)
        # self.log_stats_t = t_env

        # Store TV
        if interval_id == 1:
            self.tv_list_1.append(np.mean(tv_divs))
        elif interval_id == 2:
            # Report sum of "1"
            self.logger.log_stat(f"policy_diff/TV_sum_1", np.sum(self.tv_list_1), t_env)
            self.logger.log_stat(f"policy_diff/TV_sum1-diff2", np.sum(self.tv_list_1) - np.mean(tv_divs), t_env)
            self.tv_list_1 = []
            #
            self.tv_list_2.append(np.mean(tv_divs))
        elif interval_id == 3:
            # Report sum of "2"
            self.logger.log_stat(f"policy_diff/TV_sum_2", np.sum(self.tv_list_2), t_env)
            self.logger.log_stat(f"policy_diff/TV_sum2-diff3", np.sum(self.tv_list_2) - np.mean(tv_divs), t_env)
            self.tv_list_2 = []

        # Update previous agent
        if interval_id == 1:
            self.prev1_mac_agent_for_policy_diff = copy.deepcopy(self.mac.agent)
            self.prev1_epsilon = cur_epsilon
        elif interval_id == 2:
            self.prev2_mac_agent_for_policy_diff = copy.deepcopy(self.mac.agent)
            self.prev2_epsilon = cur_epsilon
        elif interval_id == 3:
            self.prev3_mac_agent_for_policy_diff = copy.deepcopy(self.mac.agent)
            self.prev3_epsilon = cur_epsilon

    def compute_epsilon_greedy_policy(self, batch, agent, avail_actions, epsilon, t_env):
        # Store current agent
        backup_agent = self.mac.agent
        #
        self.mac.agent = agent
        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t, dormant_record=None,
                                          obs_key=self.preprocess_manager.get_obs_key(t_env))
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Remove the last step in a episode

        masked_q_values = mac_out.clone()
        unmasked_q_values = mac_out.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")

        # Calculate probabilities
        num_actions = masked_q_values.size(-1)
        greedy_actions = masked_q_values.max(dim=3)[1]
        # Initialize probabilities (ensure each at least 1e-8 to avoid NaN)
        probabilities = th.full_like(masked_q_values, fill_value=max(1e-8 * num_actions, epsilon) / num_actions)

        # Set greedy action probability
        # greedy_actions shape: [batch_size, seq_len, n_agents]
        # masked_q_values shape: [batch_size, seq_len, n_agents, n_actions]
        batch_range = th.arange(greedy_actions.size(0)).unsqueeze(-1).unsqueeze(-1).expand(
            -1, greedy_actions.size(1), greedy_actions.size(2))
        seq_range = th.arange(greedy_actions.size(1)).unsqueeze(0).unsqueeze(-1).expand(
            greedy_actions.size(0), -1, greedy_actions.size(2))
        agent_range = th.arange(greedy_actions.size(2)).unsqueeze(0).unsqueeze(0).expand(
            greedy_actions.size(0), greedy_actions.size(1), -1)
        # Update probabilities for greedy actions
        probabilities[batch_range, seq_range, agent_range, greedy_actions] += (1 - epsilon)
        # Restore current agent
        self.mac.agent = backup_agent
        return probabilities, unmasked_q_values

    def _check_dormant_ratio(self, layer: np.ndarray, t_env, layer_name: str):
        layer_total = layer.sum()
        layer_ratio = layer / layer_total

        for tau in self.args.dormant_all_taus:
            n_dormant_neurons = np.sum(layer_ratio <= tau)
            dormant_ratio = n_dormant_neurons / layer.shape[0]
            self.logger.log_stat(f"dormant_ratio/{layer_name}_tau={tau}", dormant_ratio, t_env)

        self.logger.log_stat(f"dormant_ratio/{layer_name}_layerAbsTotal", layer_total, t_env)

    def check_dormant_neurons(self, buffer, t_env: int):
        dormant_record = {'x': [], 'h': []}

        for i_batch in range(self.args.dormant_n_batches):
            batch = self.preprocess_manager.sample_current_needed_data(buffer, self.args, t_env)

            # Truncate batch to only filled timesteps
            max_ep_t = batch.max_t_filled()
            batch = batch[:, :max_ep_t]

            if batch.device != self.args.device:
                batch.to(self.args.device)

            # ----------------------------------------------
            # Calculate estimated Q-Values
            self.mac.init_hidden(batch.batch_size)
            with th.no_grad():
                for t in range(batch.max_seq_length):
                    _ = self.mac.forward(batch, t=t, dormant_record=dormant_record,
                                         obs_key=self.preprocess_manager.get_obs_key(t_env))
        x = np.array(dormant_record['x']).mean(axis=0)
        h = np.array(dormant_record['h']).mean(axis=0)
        self._check_dormant_ratio(x, t_env, 'x')
        self._check_dormant_ratio(h, t_env, 'h')

    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        if self.mixer is not None:
            for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def reset_param(self, reset_pos_str: str):
        """Reset last layer."""
        mac_type = self.args.mac
        assert mac_type in ['non_shared_mac', 'basic_mac']
        using_ps = mac_type == 'basic_mac'

        if self.args.n_distillation_grad_steps > 0:
            self.teacher_mac_agent = copy.deepcopy(self.mac.agent)
            if self.mixer is not None:
                self.teacher_mixer = copy.deepcopy(self.mixer)
        if 'fc2' in reset_pos_str:
            if using_ps:
                self.mac.agent.fc2.apply(reset_weights)
                self.target_mac.agent.fc2.load_state_dict(self.mac.agent.fc2.state_dict())
            else:
                for agent, target_agent in zip(self.mac.agent.agents, self.target_mac.agent.agents):
                    agent.fc2.apply(reset_weights)
                    target_agent.fc2.load_state_dict(agent.fc2.state_dict())
        if 'fc1' in reset_pos_str:
            if using_ps:
                self.mac.agent.fc1.apply(reset_weights)
                self.target_mac.agent.fc1.load_state_dict(self.mac.agent.fc1.state_dict())
            else:
                for agent, target_agent in zip(self.mac.agent.agents, self.target_mac.agent.agents):
                    agent.fc1.apply(reset_weights)
                    target_agent.fc1.load_state_dict(agent.fc1.state_dict())
        if 'rnn' in reset_pos_str:
            if using_ps:
                self.mac.agent.rnn.apply(reset_weights)
                self.target_mac.agent.rnn.load_state_dict(self.mac.agent.rnn.state_dict())
            else:
                for agent, target_agent in zip(self.mac.agent.agents, self.target_mac.agent.agents):
                    agent.rnn.apply(reset_weights)
                    target_agent.rnn.load_state_dict(agent.rnn.state_dict())

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
