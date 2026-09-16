from collections import deque
from copy import deepcopy
from typing import Callable, List, Dict, Union
from typing import Optional

import numpy as np

from epy_tools.lbf_utils import lbf_preprocess
from epy_tools.resco_utils import resco_preprocess
from epy_tools.rware_utils import rware_preprocess
from epy_tools.cityflow_utils import cityflow_topo_preprocess

class NonStationaryUCB:
    def __init__(self, args, ucb_desc: str):
        """
        Preprocess desc: e.g., "lbf:15s->ucb(1s,2s,3s,4s,5s),c=1.0,w=1000"
        ucb description: such as "ucb(1s,2s,3s,4s,5s),c=1.0,w=1000" -> 1s, 2s, 3s, 4s, 5s are the arms,
        c is the exploration constant, w is the window size
        """
        try:
            str_rest = ucb_desc
            if ',add=' in ucb_desc:
                str_rest, self.return_add = str_rest.split(',add=')
            else:
                self.return_add = 0.0
            if ',div=' in ucb_desc:
                str_rest, self.return_div = str_rest.split(',div=')
            else:
                self.return_div = 1.0
            str_rest, self.window_size = str_rest.split(',w=')
            str_rest, self.c = str_rest.split('),c=')
            self.arm_names = str_rest.replace('ucb(', '').split(',')
        except:
            raise Exception(f'Invalid ucb_desc: {ucb_desc}')

        # Type
        self.c = float(self.c)
        self.window_size = int(self.window_size)
        self.return_div = float(self.return_div)
        self.return_add = float(self.return_add)
        print(
            f'[NonStationaryUCB] arm_names: {self.arm_names}, c: {self.c}, window_size: {self.window_size}, return_norm: {self.return_div}')

        #
        self.n_arms = len(self.arm_names)
        self.values = [deque(maxlen=self.window_size) for _ in range(self.n_arms)]
        self.history = deque(maxlen=self.window_size)
    '''
    def compute_each_sight_ucb_exploitation_value(self, t_env):
        return {arm_name:  (np.mean(values) + self.return_add) / self.return_div if len(values) > 0 else 0.0
                for arm_name, values in zip(self.arm_names, self.values)}
    
    '''
    def compute_each_sight_ucb_exploitation_value(self, t_env):
        return {arm_name:  min(1.0, t_env / 5e5) * (np.mean(values) + self.return_add) / self.return_div if len(values) > 0 else 0.0
                for arm_name, values in zip(self.arm_names, self.values)}
    

    
    def compute_each_sight_ucb_exploration_value(self):
        total_in_window = sum([len(values) for values in self.values])
        each_sight_exploration = {}
        for arm in range(self.n_arms):
            n = len(self.values[arm])
            if n == 0:
                each_sight_exploration[self.arm_names[arm]] = float('inf')
                continue
            each_sight_exploration[self.arm_names[arm]] = self.c * np.sqrt(np.log(total_in_window) / n)
        return each_sight_exploration

    def select_arm(self, t_env, greedy: bool = False):
        # 確保每個手臂至少選擇一次以避免除零錯誤
        total = len(self.history)
        if len(self.history) < self.n_arms:
            idx = self.arm_names[total]
            # print(f'[UCB select_arm] ucb_values: N/A (just starting), idx: {idx}')
            return idx

        # Calculate UCB values
        each_sight_exploit_values = self.compute_each_sight_ucb_exploitation_value(t_env)
        each_sight_explore_values = self.compute_each_sight_ucb_exploration_value()
        each_sight_ucb = {arm_name: exploit_value + (explore_value * (not greedy))
                          for arm_name, exploit_value, explore_value in
                          zip(self.arm_names, each_sight_exploit_values.values(), each_sight_explore_values.values())}
        ucb_values = list(each_sight_ucb.values())
        idx = np.argmax(ucb_values)

        # for arm_name in self.arm_names:
        #     print(f'{arm_name}', end='\t')
        # print()
        # for ucb_value in ucb_values:
        #     print(f'{ucb_value:.2f}', end='\t')
        # print()

        # print(f'[UCB select_arm] idx: {idx}')
        # for ucb_value in ucb_values:
        #     print(f'{ucb_value:.4f}', end=' ')
        # print()

        return self.arm_names[idx]

    def update(self, arm_name, reward):
        arm = self.arm_names.index(arm_name)
        # If window_size is full, remove the oldest value
        if len(self.history) == self.window_size:
            oldest_arm = self.history.popleft()
            self.values[oldest_arm].popleft()

        self.values[arm].append(reward)
        self.history.append(arm)

        # print(f'[UCB update] \n'
        #       f'arm: {arm_name}\n'
        #       f'reward: {reward}\n'
        #       f'values: {self.values}\n'
        #       f'history: {self.history}')

    def get_each_sight_ratio_in_window(self) -> Dict[str, float]:
        cur_n_in_window = sum([len(values) for values in self.values])
        sight_count = {arm_name: len(self.values[i]) / cur_n_in_window
                       for i, arm_name in enumerate(self.arm_names)}
        return sight_count


class PreprocessManager:
    """
    先定義一下，``preprocess_desc`` 的格式，並且用 ``"lbf:15s->2s@0,4s@1e6,6s@2e6,8s@3e6,15@4e6"`` 做為範例。
    ``preprocess_desc`` 是一個用來描述 preprocess 的字串，格式如下：
    - 若是 None，代表不需要 preprocess；如非弄則應為一個字串，以下用非 None 的狀況來講解。
    - 字串裡的冒號前面是環境類型，用變數名稱``preprocess_env_type``。在此例為 "lbf"
    - 冒號後面是一個字串，稱為 ``preprocess_schedule_str``，包含一個箭頭：
        - 箭頭前稱為``original_visibility_str``，代表原始環境的視野，此例為 "15s"
        - 箭頭後稱為``switch_visibility_str``，代表在哪些時間點要轉換視野?以及換成什麼樣的視野?
            - 若不含一些關鍵字(例如equal,arithmetic)則為手動指定時間點，以及轉換後的視野。
                - ``switch_visibility_str`` 是一個字串，用逗號分隔，每個部分都是一個轉換描述，格式為 "``visibility_str``@``t_env``"。
                  在此例中，"2s@0,4s@1e6,6s@2e6,8s@3e6,15@4e6" 代表在 t=0 時，將視野從 15s 轉換成 2s,
                  並且在 t=1e6 時，將視野從 2s 轉換成 4s, 以此類推。
            - 若含有關鍵字，則為自動設定時間點，以及轉換後的視野。
                - 例如 "lbf: 15s→equal(1s,2s,3s,4s,5s,6s,15s)~1e7"，在此的 `switch_visibility_str`
                  為 "equal(1s,2s,3s,4s,5s,6s,15s)~1e7"，代表這 7 個 sight 各自約 1e7/7 的長度。
                - 例如 "lbf: 15s→arithmetic(1s,2s,3s,4s,5s,6s,15s)~1e7"，在此的 `switch_visibility_str`
                  為 "arithmetic(1s,2s,3s,4s,5s,6s,15s)~1e7"，代表這 7 個 sight 各自的時間從小到大為
                  1*1e7/28,2*1e7/28,3*1e7/28,...,7*1e7/28 (等比數列，7 種 sight 分別權重小到大為 1,2,...,7，
                  總權重為1+2+...+7=28，所以以 2s 來說它分配的時間為 總時間*權重/總權重=1e7*2/28)。
                - 若 "~時間"後面有 "#"，表示有使用隨機的方式來 sample visibility,
                  例如："lbf: 15s→equal(1s,2s,3s,4s,5s,6s,15s)~1e7#gradualNext"。
                  "#"後面的字串是詳細的隨機法，對應的變數名稱為 ``sample_visibility_method``。



    對於 ucb 來說，key 只會用到 "obs"



    """

    available_auto_switch_visibility_str = ['equal', 'arithmetic']

    def __init__(self, n_agents: int, preprocess_desc: Optional[str], args,
                reset_after_switch_visibility: Union[bool, str] = False,
                reset_buffer_after_switch_visibility: Optional[bool] = None,
                add_sight_id_len: Optional[int] = None):
        """

        reset_manual_schedule: 讓 run.py 可以問說是否該 reset 了

        注意！這裡的"resco"適用的範圍在resco的corridor系列，其它的不適用！


        == Args ==
        - n_agents: 環境中的 agent 數量
        - preprocess_desc: 一個描述 preprocess 的字串，格式如上述
        - args: 用來做輔助用的，
        - reset_after_switch_visibility: 一個 bool 或是一個字串，用來控制何時要 reset model
            - 若是 bool，則代表是否要在每個 switch time 都 reset model
            - 若是字串，則代表何時要 reset model，格式如 ""lbf:15s->1s@0,2s@1e6,3s@2e6,4s@3e6"，代表在 t=1e6,2e6,3e6
              時要 reset model
        - add_sight_id_len: 若有的話，obs最前面一段會是這個東西，所以在preprocess時要skip這一段
        """
        self.n_agents = n_agents
        self.args = args
        self.preprocess_desc = preprocess_desc
        if self.preprocess_desc == "" or self.preprocess_desc == "None":
            self.preprocess_desc = None
        self.use_preprocess: bool = preprocess_desc is not None
        self.reset_buffer_after_switch_visibility = reset_buffer_after_switch_visibility
        self.adaptive_workers: Optional[NonStationaryUCB] = None
        self.add_sight_id_len = int(add_sight_id_len) if add_sight_id_len is not None else None
        self.cityflow_adjacency = args.cityflow_adjacency  if args.cityflow_adjacency  is not None else None

        if self.use_preprocess:
            try:
                self.preprocess_env_type, self.preprocess_schedule_str = preprocess_desc.split(':')
            except ValueError:
                raise ValueError(f'Invalid format for preprocess_desc: "{preprocess_desc}",'
                                 f' should be "``preprocess_env_type``:``preprocess_schedule_str``"')
            try:
                self.original_visibility_str, self.switch_visibility_str = self.preprocess_schedule_str.split('->')
            except ValueError:
                raise ValueError(f'Invalid format for preprocess_schedule_str: "{self.preprocess_schedule_str}",'
                                 f' should be "``original_visibility_str``->``switch_visibility_str``"')
            # For random-sampling-visibilities
            self.sample_visibility_method = None
            # Parse switch_visibility_str
            # Check if there's a keyword
            if 'ucb' in self.switch_visibility_str:
                self.switch_time_to_visibility_str = None

                # 一個 episode 切成幾段，例如 5 段
                self.temporal_ucb_bins = getattr(args, "temporal_ucb_bins", 10)

                # temporal_adaptive_workers[time_bin][agent_id]
                self.temporal_adaptive_workers = [
                    [
                        NonStationaryUCB(args=args, ucb_desc=self.switch_visibility_str)
                        for _ in range(n_agents)
                    ]
                    for _ in range(self.temporal_ucb_bins)
                ]

                # 保留 adaptive_workers，讓原本 run.py 的判斷不會壞掉
                # 這裡只拿 bin 0 當 compatibility 用
                self.adaptive_workers = self.temporal_adaptive_workers[0]

                self.possible_visibility_str = self.temporal_adaptive_workers[0][0].arm_names
            elif any([keyword in self.switch_visibility_str for keyword in self.available_auto_switch_visibility_str]):
                switch_time_to_visibility_str = self._parse_auto_switch_visibility_str()
                self.switch_time_to_visibility_str = dict(
                    sorted(switch_time_to_visibility_str.items(), key=lambda x: x[0]))  # Sort by time
                self.possible_visibility_str: List[str] = list(self.switch_time_to_visibility_str.values())
            else:
                switch_time_to_visibility_str = {}
                for visibility_str_and_time in self.switch_visibility_str.split(','):
                    switch_sight, switch_time = visibility_str_and_time.split('@')
                    switch_time = int(float(switch_time))
                    switch_time_to_visibility_str[switch_time] = switch_sight
                self.switch_time_to_visibility_str = dict(
                    sorted(switch_time_to_visibility_str.items(), key=lambda x: x[0]))  # Sort by time
                self.possible_visibility_str: List[str] = list(self.switch_time_to_visibility_str.values())
        else:
            self.preprocess_env_type, self.preprocess_schedule_str = None, None
            self.original_visibility_str, self.switch_visibility_str = None, None
            self.switch_time_to_visibility_str = None
            self.possible_visibility_str = []
        print(f'self.switch_time_to_visibility_str: {self.switch_time_to_visibility_str}')
        print(f'self.possible_visibility_str: {self.possible_visibility_str}')
        #
        self.preprocess_schedule_func = self.get_schedule_func()
        #
        self.reset_after_switch_visibility = reset_after_switch_visibility
        if isinstance(reset_after_switch_visibility, bool):
            self.reset_not_yet_switch_time = list(set(list(
                self.switch_time_to_visibility_str.keys()))) if reset_after_switch_visibility else []
        elif isinstance(reset_after_switch_visibility, str):
            self.reset_not_yet_switch_time = list(
                set([int(float(x.split('@')[-1])) for x in reset_after_switch_visibility.split('->')[-1].split(',')]))
        else:
            raise ValueError(f'Invalid type for reset_after_switch_visibility: {reset_after_switch_visibility}')
        # Remove time = 0
        if 0 in self.reset_not_yet_switch_time:
            self.reset_not_yet_switch_time.remove(0)
        self.reset_all_satisfied_t_env = {switch_time: None for switch_time in self.reset_not_yet_switch_time}

        # Time to "obs_key"
        # 這是用來記錄每個時間點對應到的 obs_key 是什麼
        # 每個時間點只對應到一個 obs_key，無論 query 幾次，同樣的 t_env 都會回傳同樣的 obs_key
        # (因為導入了 random sampling obs_key 的機制)
        self.time_to_obs_key = {}

        if self.add_sight_id_len is not None:
            if self.switch_time_to_visibility_str is not None:
                raise NotImplementedError('Not support POEM + add sight id so far')
            if len(self.possible_visibility_str) != self.add_sight_id_len:
                raise ValueError(f'len(self.possible_visibility_str) ({len(self.possible_visibility_str)}) != '
                                 f'self.add_sight_id_len ({self.add_sight_id_len})')

    @property
    def n_sights(self):
        return len(self.possible_visibility_str)

    def get_adaptive_sight_index(self, adaptive_sight):
        """Return the one-hot vector of the sight by the obs_key"""
        return self.adaptive_workers[0].arm_names.index(adaptive_sight)

    def _parse_auto_switch_visibility_str(self):
        """Parse the switch_visibility_str if it contains a keyword. Return a dict of switch time to visibility str."""
        func_str, total_time = self.switch_visibility_str.split('~')
        # Check if using random sampling visibility
        if '#' in total_time:
            total_time, self.sample_visibility_method = total_time.split('#')
        print(f'self.sample_visibility_method: {self.sample_visibility_method}')
        if self.sample_visibility_method is not None:
            assert self.reset_buffer_after_switch_visibility in [None,
                                                                 False], 'Not support both random sampling visibility and ' \
                                                                         'reset_buffer_after_switch_visibility'
        total_time = int(float(total_time))
        #
        assert self.args.t_max >= total_time, f't_max ({self.args.t_max}) should be >= total_time ({total_time})'
        func_name = func_str.split('(')[0]
        func_str = func_str.replace(f'{func_name}(', '').replace(')', '')
        func_sights = func_str.split(',')
        assert func_name in self.available_auto_switch_visibility_str, f'Unknown function name: {func_name}'
        if func_name == 'equal':
            # 等間距
            switch_time_to_visibility_str = {int(np.ceil(i * total_time / len(func_sights))): sight for i, sight in
                                             enumerate(func_sights)}
        elif func_name == 'arithmetic':
            # 等差權重
            # 1st sight: 1, 2nd sight: 2, 3rd sight: 3, ..., last sight: len(func_sights)
            # Each length: total_time * (its_weight / total_weight)
            total_weight = sum(range(1, len(func_sights) + 1))
            individual_weights = [i + 1 for i in range(len(func_sights))]
            individual_length = [total_time * (weight / total_weight) for weight in individual_weights]
            individual_start_time = [sum(individual_length[:i]) for i in range(len(individual_length))]
            switch_time_to_visibility_str = {int(np.ceil(start_time)): sight for start_time, sight in
                                             zip(individual_start_time, func_sights)}
            return switch_time_to_visibility_str
        else:
            raise ValueError(f'Unknown function name: {func_name}')
        return switch_time_to_visibility_str

    def get_schedule_func(self) -> Optional[Callable[[int], str]]:
        """
        注意，若要為新環境實作，也要考慮實作 random sampling visibility 的機制。

        使用：``self.preprocess_schedule_str``
        回傳兩個東西
        (1) function:
            - Input: current step
            - Output: ``visibility_str`` for current step or None if no preprocess is needed
        (2) a list of all possible process ``visibility_str``
        """
        if not self.use_preprocess:
            return None
        if self.adaptive_workers is not None:
            return None
        elif self.preprocess_env_type in ['cityflow','lbf', 'resco', 'rware', 'asc2', 'metadrive', 'resco2']:
            start_sight = int(self.original_visibility_str[:-1])  # e.g., 15 in "15s"
            # 把 ``visibility_str`` 取 [:-1] 是因為要把 "s" 去掉
            switch_time_to_sight = {k: int(v[:-1]) for k, v in self.switch_time_to_visibility_str.items()}

            def lbf_schedule(t_env_):
                # 這個可能有個問題是：若 schedule 亂寫，可能會用 original 而非 switch_time_to_sight 最後一個
                # ※ 註：目前這幾個環境直接用這個
                if self.sample_visibility_method is None:
                    # 看目前最多符合到哪個時間點
                    new_sight = start_sight
                    for t, sight in switch_time_to_sight.items():
                        if t_env_ >= t:
                            new_sight = sight
                        else:
                            break
                    return f'{new_sight}s'
                else:  # Random sampling visibility
                    if t_env_ in self.time_to_obs_key:
                        # print(f'[{t_env_}] uses cache => {self.time_to_obs_key[t_env_]}')
                        return self.time_to_obs_key[t_env_]
                    if self.sample_visibility_method in ['gradualNext', 'gradualAll']:
                        cur_sight = None
                        next_sight = None
                        next_t = None
                        cur_start_t = None
                        cur_i = None
                        for i, (t, sight) in enumerate(switch_time_to_sight.items()):
                            if t_env_ >= t:
                                cur_sight = sight
                                cur_start_t = t
                                cur_i = i
                            else:
                                next_sight = sight  # 若在表上還有下一個但未到的，則 next_sight 為其
                                next_t = t
                                break
                        assert cur_sight is not None, (f'cur_sight is None, t_env_: {t_env_}. The schedule ('
                                                       f'{self.preprocess_schedule_str}) may be wrong.')
                        if self.sample_visibility_method == 'gradualNext':
                            # Sample from current (new_sight) and next sight (next_sight)
                            if next_sight is None:
                                selected_sight = cur_sight
                                # print(f'[{t_env_}] no next sight, use cur_sight ', end='')
                            else:  # Sample from cur_sight and next_sight
                                # 選擇 cur_sight 的機率為 1 - (t_env_ - cur_start_t) / (next_t - cur_start_t)
                                # 選擇 next_sight 的機率為 (t_env_ - cur_start_t) / (next_t - cur_start_t)
                                prob = (t_env_ - cur_start_t) / (next_t - cur_start_t)
                                selected_sight = cur_sight if np.random.rand() > prob else next_sight
                                # print(f'[{t_env_}] cur_sight: {cur_sight}, next_sight: {next_sight}, '
                                #       f'cur_start_t: {cur_start_t}, next_t: {next_t}, next_prob: {prob} ', end='')
                        elif self.sample_visibility_method == 'gradualAll':
                            raise NotImplementedError()
                            # if next_sight is None:
                            #     selected_sight = cur_sight
                            # else:
                            #     # (1) sample to check select cur or not
                            #     # (2) Uniformly sample from all sights after cur_sight
                            #     prob = (t_env_ - cur_start_t) / (next_t - cur_start_t)  # prob to select cur_sight
                            #     selected_sight = cur_sight if np.random.rand() > prob else next_sight
                        else:
                            raise ValueError(f'Unknown sample_visibility_method: {self.sample_visibility_method}')
                    else:
                        raise ValueError(f'Unknown sample_visibility_method: {self.sample_visibility_method}')
                    self.time_to_obs_key[t_env_] = f'{selected_sight}s'
                    # print(f'=> {self.time_to_obs_key[t_env_]}')
                    return self.time_to_obs_key[t_env_]  # Return the selected sight

            return lbf_schedule

        else:
            raise self._unknown_env_type_error()

    def try_get_adaptive_sights(self, t_env, greedy=False):
        # 回傳所有 agent 的距離決策 list
        return [w.select_arm(t_env=t_env, greedy=greedy) for w in self.adaptive_workers]

    def update_adaptive_agents(self, arm_names, rewards):
        # arm_names: list of strings, rewards: list of floats
        for worker, arm, rew in zip(self.adaptive_workers, arm_names, rewards):
            worker.update(arm, rew)

    def get_temporal_ucb_bin(self, t_ep, episode_limit):
        """
        根據 episode 內的時間 t_ep 決定目前是第幾個 time bin。
        例如 temporal_ucb_bins=5:
        0%~20%   -> bin 0
        20%~40%  -> bin 1
        ...
        """
        if not hasattr(self, "temporal_ucb_bins"):
            return 0

        ratio = t_ep / max(episode_limit, 1)
        time_bin = int(ratio * self.temporal_ucb_bins)
        return min(time_bin, self.temporal_ucb_bins - 1)


    def try_get_temporal_adaptive_sights(self, t_env, t_ep, episode_limit, greedy=False):
        """
        回傳目前 time_bin 下，每個 agent 選到的 sight。
        """
        time_bin = self.get_temporal_ucb_bin(t_ep, episode_limit)

        if hasattr(self, "temporal_adaptive_workers"):
            workers = self.temporal_adaptive_workers[time_bin]
        else:
            workers = self.adaptive_workers

        sights = [w.select_arm(t_env=t_env, greedy=greedy) for w in workers]
        return time_bin, sights


    def update_temporal_adaptive_agents(self, time_bin, arm_names, rewards):
        """
        更新某個 time_bin 裡面，每個 agent 的 SW-UCB。
        arm_names: list[str], 每個 agent 當時選到的 sight
        rewards: list[float], 每個 agent 的 reward-to-go
        """
        if hasattr(self, "temporal_adaptive_workers"):
            workers = self.temporal_adaptive_workers[time_bin]
        else:
            workers = self.adaptive_workers

        for worker, arm, rew in zip(workers, arm_names, rewards):
            worker.update(arm, rew)

    def get_next_visibility_str(self, t_env: int) -> Optional[str]:
        """回傳下一個視野是什麼，若不存在則回傳 None"""
        if not self.use_preprocess:
            return None
        if self.adaptive_workers is not None:
            raise ValueError('Not support adaptive_worker')
        cur_visibility_str = self.preprocess_schedule_func(t_env)
        if self.preprocess_env_type in ['cityflow','lbf', 'rware', 'asc2', 'metadrive']:
            # E.g., self.possible_visibility_str = ['2s', '4s', '6s', '8s', '15s']
            #       If cur_visibility_str = '4s', then return '6s'
            #       If cur_visibility_str = '15s', then return None
            if cur_visibility_str not in self.possible_visibility_str:
                raise ValueError(f'Unknown visibility: {cur_visibility_str}')
            cur_idx = self.possible_visibility_str.index(cur_visibility_str)
            if cur_idx == len(self.possible_visibility_str) - 1:
                return None
            return self.possible_visibility_str[cur_idx + 1]
        else:
            raise self._unknown_env_type_error()

    def get_one_preprocessed_obs(self, obs, visible_str):
        return self._get_one_preprocessed_obs(obs, visible_str)
    
    def parse_sight_value(self, s):
        # 如果已經是 int 或 float 則直接回傳
        if isinstance(s, (int, float)):
            return int(s)
        # 如果是字串，過濾掉所有非數字字元 (例如 '100m' -> '100', '0s' -> '0')
        import re
        numeric_part = re.sub(r'[^0-9]', '', str(s))
        if numeric_part == '':
            return 0 # 或者拋出更有意義的錯誤
        return int(numeric_part)

    def _get_one_preprocessed_obs(self, obs, visible_str):
        obs = np.array(obs)
        if self.add_sight_id_len is not None:
            obs_to_preprocess = obs[:, self.add_sight_id_len:]
        else:
            obs_to_preprocess = obs

        if self.preprocess_env_type == 'lbf':
            processed = lbf_preprocess(original_sight=int(self.original_visibility_str[:-1]),
                                       new_sight=int(visible_str[:-1]), original_obs=obs_to_preprocess,
                                       n_agents=self.n_agents, n_fruits=None, verbose=False)
        elif self.preprocess_env_type == 'rware':
            processed = rware_preprocess(original_sight=int(self.original_visibility_str[:-1]),
                                         new_sight=int(visible_str[:-1]), original_obs_list=obs_to_preprocess,
                                         verbose=False)
        elif self.preprocess_env_type == 'resco':
            processed = resco_preprocess(original_sight=int(self.original_visibility_str[:-1]),
                                         new_sight=int(visible_str[:-1]), original_obs=obs_to_preprocess,
                                         n_agents=self.n_agents, args=self.args)
        elif self.preprocess_env_type == 'cityflow':
            # 判斷 visible_str 是單一字串還是 List
            if isinstance(visible_str, list):
                sights = [self.parse_sight_value(s) for s in visible_str]
            else:
                sights = self.parse_sight_value(visible_str)

            processed = cityflow_topo_preprocess(
                            original_obs=obs_to_preprocess,
                            use_dsr=self.use_preprocess,
                            sight=sights) # 傳入 list 或單一 int
        elif self.preprocess_env_type == 'asc2':
            return obs
        elif self.preprocess_env_type == 'metadrive':
            return obs
        elif self.preprocess_env_type == 'resco2':
            return obs
        else:
            raise self._unknown_env_type_error()

        if self.add_sight_id_len is not None:
            processed = np.concatenate([obs[:, :self.add_sight_id_len], processed], axis=-1)
        return processed

    def get_preprocessed_obs_dict(self,
                                  step_obs: List[np.ndarray],
                                  adaptive_sight: Optional[str]) -> Dict[str, List[np.ndarray]]:
        """
        回傳：
        - 一個 dict, key 是 obs name, value 是 preprocess 過後的 obs

        :param step_obs: [bs, n_steps_per_episode, n_agents, obs_dim]
        :param adaptive_sight: 若不為 None，則回傳 adaptive_sight 的 preprocess 過後的 obs
        """
        obs = step_obs
        if not self.use_preprocess:
            """Return a function that like the normal save function。回傳: {'obs': obs, ...}"""
            return {'obs': obs}
        if adaptive_sight is not None:  # adaptive_sight only uses "obs"
            return {f'obs': self._get_one_preprocessed_obs(obs, adaptive_sight)}
        diff_visibility_obs = {}
        for visible_str in self.possible_visibility_str:
            new_obs = self._get_one_preprocessed_obs(obs, visible_str)
            diff_visibility_obs[f'obs_{visible_str}'] = new_obs
        if self.original_visibility_str not in self.possible_visibility_str:
            diff_visibility_obs[f'obs_{self.original_visibility_str}'] = deepcopy(step_obs)
        return diff_visibility_obs

    def get_obs_key(self, t_env: Optional[int] = None, visibility_str: Optional[str] = None) -> str:
        """回傳一個 key, 表示此時若要用 obs 的話，應該用 obs_dict 哪個 saved 的 key 的 obs"""
        if self.preprocess_schedule_func is None:
            return 'obs'
        assert (t_env is None) != (visibility_str is None), "At least one of it should be given."
        if t_env is not None:
            visibility_str = self.preprocess_schedule_func(t_env)
        return f'obs_{visibility_str}'

    @staticmethod
    def get_state_by_concatenate_obs_in_obs_dict(obs_dict, obs_key=None):
        """假設 state 都是由所有 agent obs 直接串接而成；用 obs_dict 內的 'obs' 來串接"""
        # 假設 obs_dict 只有一個 key: 'obs'
        if obs_key is None or obs_key == 'obs':
            assert len(obs_dict) == 1 and 'obs' in obs_dict, f'obs_dict: {obs_dict}'
            return np.concatenate(obs_dict['obs'], axis=-1)
        else:  # POEM
            return np.concatenate(obs_dict[obs_key], axis=-1)

    def sample_current_needed_data(self, buffer, args, t_env):
        """Sample the needed data from the buffer, 依此時需要怎樣的 visibility。
        原理：原本是要 sample "obs", 假設現在要 sample "obs_2s"，則把 "obs" key 更新成對應 "obs_2s" 的 obs。
        """
        # Obs: [bs, n_steps_per_episode, n_agents, obs_dim]
        # LBF obs: [bs, n_steps_per_episode, n_agents, n_fruits * 3 + n_agents * 3]
        episode_sample = buffer.sample(args.batch_size)
        return episode_sample
        # if not self.use_preprocess or args.turn_off_train_preprocess:
        #     # with open(f"kkk_.txt", "a") as f:
        #     #     f.write(f"{episode_sample['obs'][0][0][0].cpu().detach().numpy().tolist()}\n")
        #     # with open(f"kkk_.txt", "r") as f:
        #     #     if len(f.readlines()) > 100000:
        #     #         raise ValueError("Too many lines in file 'jjj.txt'")
        #     return episode_sample
        # obs_key = self.get_current_obs_key(t_env)
        # episode_sample.update({'obs': deepcopy(episode_sample[obs_key])})
        # # print(f'obs   : {episode_sample["obs"][0][0][0].cpu().numpy().tolist()}')
        # # print(f'obs_2s: {episode_sample["obs_2s"][0][0][0].cpu().numpy().tolist()}')
        #
        # # 將 ``batch_for_select_action['obs'][0] 存到檔案 "jjj.txt" 中
        # # 若存在，則append (一行一行)
        # # with open(f"kkk_{obs_key}.txt", "a") as f:
        # #     f.write(f"{episode_sample['obs'][0][0][0].cpu().detach().numpy().tolist()}\n")
        # # with open(f"kkk_{obs_key}.txt", "r") as f:
        # #     if len(f.readlines()) > 100000:
        # #         raise ValueError("Too many lines in file 'jjj.txt'")
        #
        # return episode_sample

    def get_visibility_scalar_to_log(self, t_env: Optional[int] = None, visibility_str: Optional[str] = None
                                     ) -> Optional[Union[int, float]]:
        """訓練過程我們可能想知道目前的視野是多少，根據環境，回傳一個 scalar 以供 log。
        也可以直接給 visibility str。
        """
        if not self.use_preprocess:
            return None
        if self.adaptive_workers is not None:
            return None
        assert (t_env is None) != (visibility_str is None), "At least one of it should be given."
        if t_env is not None:
            visibility_str = self.preprocess_schedule_func(t_env)
        if self.preprocess_env_type in ['lbf', 'resco', 'rware', 'asc2', 'metadrive', 'resco2']:
            return int(visibility_str[:-1])
        else:
            raise self._unknown_env_type_error()

    def _unknown_env_type_error(self):
        return ValueError(f'Unknown env type: {self.preprocess_env_type}')

    def get_batch_for_select_action_and_update_cur_batch_for_save(self,
                                                                  state,
                                                                  avail_actions,
                                                                  original_obs,
                                                                  t_env,
                                                                  t,
                                                                  cur_batch):
        # Preprocess obs if needed
        obs_dict = self.get_preprocessed_obs_dict(original_obs)
        obs_key = self.get_obs_key(t_env=t_env)
        # Make data for select_action and save_data
        pre_data_for_save = {"state": [state],
                             "avail_actions": [avail_actions]}
        pre_data_for_select_action = deepcopy(pre_data_for_save)
        pre_data_for_select_action.update({'obs': [deepcopy(obs_dict[obs_key])]})
        pre_data_for_save.update({k: [v] for k, v in obs_dict.items()})
        # Make batch
        batch_copied_for_action_select = deepcopy(cur_batch)
        batch_copied_for_action_select.update(pre_data_for_select_action, ts=t)
        cur_batch.update(pre_data_for_save, ts=t)
        return batch_copied_for_action_select

    def is_time_to_reset_model(self, t_env) -> bool:
        """假設你 call 這個函數若拿到 True，則輸入此 t_env 就會回傳 True。"""
        if isinstance(self.reset_after_switch_visibility, (str, bool)):
            # print('-----------------------------------------------')
            # print(f't_env: {t_env}, self.reset_not_yet_switch_time: {self.reset_not_yet_switch_time}')
            # print(f'self.reset_all_satisfied_t_env: {self.reset_all_satisfied_t_env}')
            if t_env in self.reset_all_satisfied_t_env.values():
                return True
            if len(self.reset_not_yet_switch_time) > 0:
                if t_env >= self.reset_not_yet_switch_time[0]:
                    self.reset_all_satisfied_t_env[self.reset_not_yet_switch_time[0]] = t_env
                    self.reset_not_yet_switch_time.pop(0)
                    return True
        return False
