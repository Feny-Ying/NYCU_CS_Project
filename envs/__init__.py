import os
import sys
from functools import partial
from typing import Optional,List

import gym
import numpy as np
import pretrained
from absl import logging
from gym import ObservationWrapper, spaces
from gym.spaces import flatdim
from gym.wrappers import TimeLimit as GymTimeLimit

from cityflow import Engine
import json
import math

from smac.env import MultiAgentEnv, StarCraft2Env


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


class AdaptiveSightStarCraft2Env(StarCraft2Env):
    """Override the unit_sight_range method to adapt to the adaptive sight.

    各 agent 的 dist 資訊仍除以 DEFAULT_SIGHT(保持絕對數值)
    但 obs 多一個元素，表示當前的 sight，該值為 `cur_sight / DEFAULT_SIGHT`
    DEFAULT_SIGHT = 9

    Note: Not support add sight one-hot yet.
    """
    DEFAULT_SIGHT = 9
    DEFAULT_SHOOT_RANGE = 6

    def __init__(self, add_sight_id_len: Optional[bool] = None,
                 no_cur_sight_info=False, shoot_range: Optional[int] = None,
                 concatenate_obs_as_state: bool = False,
                 dsr_not_influence_concat_state: bool = False,
                 dsr_not_influence_obs: bool = False,
                 default_sight_range: int = DEFAULT_SIGHT,
                 # 對於 ConcatenateObsAsState 的情況，分析 state 組成的時 concat 多少 agent 的 obs (令 SR=inf)
                 #             None: 不使用
                 #             int: 使用 int 個 agent 的 obs。其它 mask 成零。
                 when_obs_instead_of_state_leave_how_many_ag_obs: Optional[int] = None,
                 # Others
                 **kwargs):
        if add_sight_id_len:
            if no_cur_sight_info is not True:
                raise ValueError('If add_sight_id_len is True, no_cur_sight_info must be False')
        super().__init__(**kwargs)
        print(f'self.episode_limit = {self.episode_limit}')

        self.dsr_not_influence_concat_state = dsr_not_influence_concat_state
        self.dsr_not_influence_obs = dsr_not_influence_obs
        # if self.obs_instead_of_state is not True:
        #     if self.dsr_not_influence_concat_state is False:
        #         raise ValueError('If obs_instead_of_state is False, dsr_not_influence_concat_state must be True')

        self.add_sight_id_len = add_sight_id_len
        self.cur_adaptive_sight = None
        self.cur_adaptive_sight_idx = None
        self.no_cur_sight_info = no_cur_sight_info
        self.shoot_range = shoot_range
        self.concatenate_obs_as_state = concatenate_obs_as_state
        if self.concatenate_obs_as_state:
            raise ValueError('This is deprecated now.')
        self.DEFAULT_SIGHT = int(default_sight_range)
        print(f'Default sight range = {self.DEFAULT_SIGHT}')
        self.when_obs_instead_of_state_leave_how_many_ag_obs = when_obs_instead_of_state_leave_how_many_ag_obs
        if self.when_obs_instead_of_state_leave_how_many_ag_obs is not None:
            if self.obs_instead_of_state is not True:
                raise ValueError('If when_obs_instead_of_state_leave_how_many_ag_obs is not None, '
                                 'obs_instead_of_state must be True')
            if not (1 <= self.when_obs_instead_of_state_leave_how_many_ag_obs <= self.n_agents):
                raise ValueError('when_obs_instead_of_state_leave_how_many_ag_obs should be in [1, n_agents]')

    def unit_shoot_range(self, agent_id):
        """Returns the shooting range for an agent."""
        if self.shoot_range is not None:
            return self.shoot_range
        return self.DEFAULT_SHOOT_RANGE

    def set_cur_adaptive_sight_idx(self, cur_adaptive_sight_idx):
        self.cur_adaptive_sight_idx = cur_adaptive_sight_idx

    def set_adaptive_sight(self, adaptive_sight):
        if adaptive_sight:
            self.cur_adaptive_sight = adaptive_sight
        else:
            self.cur_adaptive_sight = self.DEFAULT_SIGHT

    def unit_sight_range(self, agent_id) -> int:
        """This returns the current sight."""
        raise ValueError('This method should not be called.')
        # if self.cur_adaptive_sight is None:
        #     raise ValueError('Please provide self.cur_adaptive_sight in advance')
        # return self.cur_adaptive_sight

    def obs_sight_range(self, agent_id) -> int:
        """This returns the current sight."""
        if self.cur_adaptive_sight is None:
            raise ValueError('Please provide self.cur_adaptive_sight in advance')
        if self.dsr_not_influence_obs:
            return self.DEFAULT_SIGHT
        else:
            return self.cur_adaptive_sight

    def state_sight_range(self):
        """當 obs_instead_of_state，時 global state 看的 obs sight 用這個。"""
        if self.cur_adaptive_sight is None:
            raise ValueError('Please provide self.cur_adaptive_sight in advance')
        if self.dsr_not_influence_concat_state:
            return self.DEFAULT_SIGHT
        else:
            return self.cur_adaptive_sight

    def reset(self):
        obs, state = super().reset()
        # print(f'AdaptiveSightStarCraft2Env reset obs: \n{obs}')
        return obs, state

    def get_visibility_matrix(self):
        """This needs no modification because it doesn't do normalization."""
        return super().get_visibility_matrix()

    def get_original_obs_agent(self, agent_id, state_view: bool):

        """改成 normalizing dist 固定用 DEFAULT_SIGHT，但介定是否看得到用的是 cur_sight，而且多一個元素顯示 cur_sight

        原本的 doc 請到 parent class 看。
        """
        unit = self.get_unit_by_id(agent_id)

        move_feats_dim = self.get_obs_move_feats_size()
        enemy_feats_dim = self.get_obs_enemy_feats_size()
        ally_feats_dim = self.get_obs_ally_feats_size()
        own_feats_dim = self.get_obs_own_feats_size()

        move_feats = np.zeros(move_feats_dim, dtype=np.float32)
        enemy_feats = np.zeros(enemy_feats_dim, dtype=np.float32)
        ally_feats = np.zeros(ally_feats_dim, dtype=np.float32)
        own_feats = np.zeros(own_feats_dim, dtype=np.float32)

        if unit.health > 0:  # otherwise dead, return all zeros
            x = unit.pos.x
            y = unit.pos.y
            if state_view:
                sight_range = self.state_sight_range()
                # print(f'using state_sight_range = {sight_range}')
            else:
                sight_range = self.obs_sight_range(agent_id)
                # print(f'using obs_sight_range = {sight_range}')

            # Movement features
            avail_actions = self.get_avail_agent_actions(agent_id)
            for m in range(self.n_actions_move):
                move_feats[m] = avail_actions[m + 2]

            ind = self.n_actions_move

            if self.obs_pathing_grid:
                move_feats[
                ind: ind + self.n_obs_pathing  # noqa
                ] = self.get_surrounding_pathing(unit)
                ind += self.n_obs_pathing

            if self.obs_terrain_height:
                move_feats[ind:] = self.get_surrounding_height(unit)

            # Enemy features
            for e_id, e_unit in self.enemies.items():
                e_x = e_unit.pos.x
                e_y = e_unit.pos.y
                dist = self.distance(x, y, e_x, e_y)

                if (
                        dist < sight_range and e_unit.health > 0
                ):  # visible and alive
                    # Sight range > shoot range
                    enemy_feats[e_id, 0] = avail_actions[
                        self.n_actions_no_attack + e_id
                        ]  # available
                    enemy_feats[e_id, 1] = dist / self.DEFAULT_SIGHT  # distance
                    enemy_feats[e_id, 2] = (
                                                   e_x - x
                                           ) / self.DEFAULT_SIGHT  # relative X
                    enemy_feats[e_id, 3] = (
                                                   e_y - y
                                           ) / self.DEFAULT_SIGHT  # relative Y

                    ind = 4
                    if self.obs_all_health:
                        enemy_feats[e_id, ind] = (
                                e_unit.health / e_unit.health_max
                        )  # health
                        ind += 1
                        if self.shield_bits_enemy > 0:
                            max_shield = self.unit_max_shield(e_unit)
                            enemy_feats[e_id, ind] = (
                                    e_unit.shield / max_shield
                            )  # shield
                            ind += 1

                    if self.unit_type_bits > 0:
                        type_id = self.get_unit_type_id(e_unit, False)
                        enemy_feats[e_id, ind + type_id] = 1  # unit type

            # Ally features
            al_ids = [
                al_id for al_id in range(self.n_agents) if al_id != agent_id
            ]
            for i, al_id in enumerate(al_ids):

                al_unit = self.get_unit_by_id(al_id)
                al_x = al_unit.pos.x
                al_y = al_unit.pos.y
                dist = self.distance(x, y, al_x, al_y)

                if (
                        dist < sight_range and al_unit.health > 0
                ):  # visible and alive
                    ally_feats[i, 0] = 1  # visible
                    ally_feats[i, 1] = dist / self.DEFAULT_SIGHT  # distance
                    ally_feats[i, 2] = (al_x - x) / self.DEFAULT_SIGHT
                    ally_feats[i, 3] = (al_y - y) / self.DEFAULT_SIGHT

                    ind = 4
                    if self.obs_all_health:
                        ally_feats[i, ind] = (
                                al_unit.health / al_unit.health_max
                        )  # health
                        ind += 1
                        if self.shield_bits_ally > 0:
                            max_shield = self.unit_max_shield(al_unit)
                            ally_feats[i, ind] = (
                                    al_unit.shield / max_shield
                            )  # shield
                            ind += 1

                    if self.unit_type_bits > 0:
                        type_id = self.get_unit_type_id(al_unit, True)
                        ally_feats[i, ind + type_id] = 1
                        ind += self.unit_type_bits

                    if self.obs_last_action:
                        ally_feats[i, ind:] = self.last_action[al_id]

            # Own features
            ind = 0
            if self.obs_own_health:
                own_feats[ind] = unit.health / unit.health_max
                ind += 1
                if self.shield_bits_ally > 0:
                    max_shield = self.unit_max_shield(unit)
                    own_feats[ind] = unit.shield / max_shield
                    ind += 1

            if self.unit_type_bits > 0:
                type_id = self.get_unit_type_id(unit, True)
                own_feats[ind + type_id] = 1

        agent_obs = np.concatenate(
            (
                move_feats.flatten(),
                enemy_feats.flatten(),
                ally_feats.flatten(),
                own_feats.flatten(),
            )
        )

        if self.obs_timestep_number:
            agent_obs = np.append(
                agent_obs, self._episode_steps / self.episode_limit
            )

        if self.debug:
            logging.debug("Obs Agent: {}".format(agent_id).center(60, "-"))
            logging.debug(
                "Avail. actions {}".format(
                    self.get_avail_agent_actions(agent_id)
                )
            )
            logging.debug("Move feats {}".format(move_feats))
            logging.debug("Enemy feats {}".format(enemy_feats))
            logging.debug("Ally feats {}".format(ally_feats))
            logging.debug("Own feats {}".format(own_feats))

        return agent_obs

    def add_sight_info(self, obs_or_state):
        """Add sight info to obs or state."""
        if not self.no_cur_sight_info:
            sight_info_scalar = float(self.cur_adaptive_sight / self.DEFAULT_SIGHT)
            obs_or_state = np.append(obs_or_state, sight_info_scalar)
        elif self.add_sight_id_len:
            assert self.cur_adaptive_sight_idx is not None, 'Please provide self.cur_adaptive_sight_idx in advance'
            obs_or_state = np.append(obs_or_state, np.eye(self.add_sight_id_len)[self.cur_adaptive_sight_idx])
        return obs_or_state

    def get_obs_agent(self, agent_id):
        raise ValueError('This method should not be called. Use get_obs_agent_given_whether_state_view instead.')

    def get_obs_agent_given_whether_state_view(self, agent_id, state_view):
        """先拿到原始的 obs，再加上 sight info 若需要的話。"""
        original_obs = self.get_original_obs_agent(agent_id, state_view)
        agent_obs = self.add_sight_info(original_obs)
        # print(f'[ id = {agent_id} ] Shape of original_obs = {original_obs.shape}, '
        #       f'Shape of agent_obs = {agent_obs.shape}')
        return agent_obs

    def get_obs(self, ):
        """Returns all agent observations in a list.
        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        agents_obs = [self.get_obs_agent_given_whether_state_view(i, state_view=False) for i in range(self.n_agents)]
        return agents_obs

    def get_state(self):
        """Returns the global state.
        NOTE: This functon should not be used during decentralised execution.
        """
        # ----- Concatenate obs as state -----
        if self.obs_instead_of_state:
            if self.when_obs_instead_of_state_leave_how_many_ag_obs is not None:
                # 假設原本有 N 個 agent，現在只用前 M 個 agent 的 obs 來做 state，obs size = L
                # 那麼原本會是 L * N，現在是 L * N 的前 L * M 是有東西的，後面都要填 0
                # 所以你要先拿到原本的 N 個人的 obs 再把後面 N-M 個人的 obs 都填 0
                agents_obs = [self.get_original_obs_agent(i, state_view=True) for i in range(self.n_agents)]
                obs_size = agents_obs[0].shape[0]
                obs_concat = np.concatenate(agents_obs, axis=0).astype(np.float32)
                state = np.concatenate(
                    [obs_concat[:self.when_obs_instead_of_state_leave_how_many_ag_obs * obs_size],
                     np.zeros((self.n_agents - self.when_obs_instead_of_state_leave_how_many_ag_obs) * obs_size)])
            else:  # Normal case
                agents_obs = [self.get_original_obs_agent(i, state_view=True) for i in range(self.n_agents)]
                obs_concat = np.concatenate(agents_obs, axis=0).astype(np.float32)
                state = obs_concat
        # ----- Use original state -----
        else:
            state_dict = self.get_state_dict()

            state = np.append(
                state_dict["allies"].flatten(), state_dict["enemies"].flatten()
            )
            if "last_action" in state_dict:
                state = np.append(state, state_dict["last_action"].flatten())
            if "timestep" in state_dict:
                state = np.append(state, state_dict["timestep"])

            state = state.astype(dtype=np.float32)

            if self.debug:
                logging.debug("STATE".center(60, "-"))
                logging.debug("Ally state {}".format(state_dict["allies"]))
                logging.debug("Enemy state {}".format(state_dict["enemies"]))
                if self.state_last_action:
                    logging.debug("Last actions {}".format(self.last_action))

        # ----- Add sight info -----
        real_state = self.add_sight_info(state)
        # print(f'Shape of states before/after add sight info: {state.shape} / {real_state.shape}')
        return real_state

    def get_original_obs_size(self):
        """Returns the size of the observation."""
        own_feats = self.get_obs_own_feats_size()
        move_feats = self.get_obs_move_feats_size()

        n_enemies, n_enemy_feats = self.get_obs_enemy_feats_size()
        n_allies, n_ally_feats = self.get_obs_ally_feats_size()

        enemy_feats = n_enemies * n_enemy_feats
        ally_feats = n_allies * n_ally_feats
        print(f'original_obs_size = {move_feats + enemy_feats + ally_feats + own_feats}')
        return move_feats + enemy_feats + ally_feats + own_feats

    def get_original_state_size(self):
        """Returns the size of the global state."""
        if self.obs_instead_of_state:
            return self.get_original_obs_size() * self.n_agents

        nf_al = 4 + self.shield_bits_ally + self.unit_type_bits
        nf_en = 3 + self.shield_bits_enemy + self.unit_type_bits

        enemy_state = self.n_enemies * nf_en
        ally_state = self.n_agents * nf_al

        size = enemy_state + ally_state

        if self.state_last_action:
            size += self.n_agents * self.n_actions
        if self.state_timestep_number:
            size += 1
        print(f'original_state_size = {size}')
        return size

    def get_state_size(self):
        """Returns the size of the global state.
        註：如要加 sight id or sight scalar，state 也要加，不然 global value function 會不知道。
        並且，若 obs_instead_of_state 且要加 sight info，用不含 sight info 的 obs 來做 state 再加上 sight info。
        """
        if self.concatenate_obs_as_state:
            # return self.get_obs_size() * self.n_agents
            raise ValueError('This is deprecated now.')

        original_state_size = self.get_original_state_size()
        if not self.no_cur_sight_info:
            state_size = original_state_size + 1
        elif self.add_sight_id_len:
            state_size = original_state_size + self.add_sight_id_len
        else:
            state_size = original_state_size
        print(f'self.obs_instead_of_state = {self.obs_instead_of_state}, '
              f'no_cur_sight_info = {self.no_cur_sight_info}, add_sight_id_len = {self.add_sight_id_len}')
        print(f'original_state_size = {original_state_size}, real state_size = {state_size}')
        return state_size

    def get_obs_size(self):
        """Returns the size of the observation."""
        original_obs_size = self.get_original_obs_size()
        if not self.no_cur_sight_info:  # Use sight_info_scalar
            obs_size = original_obs_size + 1
        elif self.add_sight_id_len:  # Use sight_info one-hot
            obs_size = original_obs_size + self.add_sight_id_len
        else:  # None of them is used
            obs_size = original_obs_size
        print(f'original_obs_size = {original_obs_size}, real obs_size = {obs_size}')
        return obs_size

    def get_stats(self):
        # 目前的資訊從 info 拿，為避免拿到重複的，這裡 SC2 不拿 get_stats
        return {}


REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["asc2"] = partial(env_fn, env=AdaptiveSightStarCraft2Env)

if sys.platform == "linux":
    os.environ.setdefault(
        "SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII")
    )


class TimeLimit(GymTimeLimit):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        # if self.env.spec is not None:
        #     self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert (
                self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not all(done) \
                if type(done) is list \
                else not done
            done = len(observation) * [True]
        return observation, reward, done, info


class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation of individual agents."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)

        ma_spaces = []

        for sa_obs in env.observation_space:
            flatdim = spaces.flatdim(sa_obs)
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]

        self.observation_space = spaces.Tuple(tuple(ma_spaces))

    def observation(self, observation):
        return tuple(
            [
                spaces.flatten(obs_space, obs)
                for obs_space, obs in zip(self.env.observation_space, observation)
            ]
        )


class AddAdaptiveSight(ObservationWrapper):
    r"""Observation wrapper that flattens the observation of individual agents."""

    def __init__(self, env, add_sight_id_len: int):
        super(AddAdaptiveSight, self).__init__(env)

        ma_spaces = []

        for sa_obs in env.observation_space:
            flatdim = spaces.flatdim(sa_obs)
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim + add_sight_id_len,),
                    dtype=np.float32,
                )
            ]

        self.observation_space = spaces.Tuple(tuple(ma_spaces))
        self.add_sight_id_len = add_sight_id_len

        self.cur_adaptive_sight_idx = None

    def observation(self, observation):
        # According to self.cur_adaptive_sight_idx, add sight id into observation (one-hot) for each agent
        # Add in the front
        assert self.cur_adaptive_sight_idx is not None, 'Please provide self.cur_adaptive_sight_idx in advance'
        return tuple([
            np.concatenate([np.eye(self.add_sight_id_len)[self.cur_adaptive_sight_idx], obs])
            for obs in observation
        ])


class _GymmaWrapper(MultiAgentEnv):
    def __init__(self, key, time_limit, pretrained_wrapper, seed, add_sight_id_len: Optional[int], **kwargs):
        self.key = key
        for _ in range(5):
            print(f'_GymmaWrapper obs_sight_id = {add_sight_id_len}')
        self.original_env = gym.make(f"{key}", **kwargs)
        self.episode_limit = time_limit
        self._env = TimeLimit(self.original_env, max_episode_steps=time_limit)
        self._env = FlattenObservation(self._env)

        self.add_sight_id_len = int(add_sight_id_len) if add_sight_id_len is not None else None
        if self.add_sight_id_len is not None:
            self._env = AddAdaptiveSight(self._env, self.add_sight_id_len)

        if pretrained_wrapper:
            self._env = getattr(pretrained, pretrained_wrapper)(self._env)

        self.n_agents = self._env.n_agents
        self._obs = None
        self._info = None

        self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(
            self._env.observation_space, key=lambda x: x.shape
        )

        self._seed = seed
        self._env.seed(self._seed)

    def set_cur_adaptive_sight_idx(self, cur_adaptive_sight_idx):
        assert self.add_sight_id_len is not None and cur_adaptive_sight_idx < self.add_sight_id_len
        self._env.cur_adaptive_sight_idx = cur_adaptive_sight_idx

    def step(self, actions):
        """ Returns reward, terminated, info """
        actions = [int(a) for a in actions]
        self._obs, reward, done, self._info = self._env.step(actions)
        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in self._obs
        ]

        if type(reward) is list:
            reward = sum(reward)
        if type(done) is list:
            done = all(done)
        return float(reward), done, {}

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self._obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise self._obs[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return flatdim(self.longest_observation_space)

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """ Returns the shape of the state"""
        if hasattr(self.original_env, 'state_size'):
            return self.original_env.state_size
        return self.n_agents * flatdim(self.longest_observation_space)

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        valid = flatdim(self._env.action_space[agent_id]) * [1]
        invalid = [0] * (self.longest_action_space.n - len(valid))
        return valid + invalid

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return flatdim(self.longest_action_space)

    def reset(self):
        """ Returns initial observations and states"""
        self._obs = self._env.reset()
        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in self._obs
        ]
        return self.get_obs(), self.get_state()

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    def seed(self):
        return self._env.seed

    def save_replay(self):
        pass

    def get_stats(self):
        return {}


class _MyGymmaWrapper(_GymmaWrapper):
    """修改：讓 info 不是空集合，而含有遊戲的結果。以用來 log。"""

    def get_stats(self):
        stats = self.original_env.get_stats()
        return stats

    def set_adaptive_sight(self, adaptive_sight):
        self.original_env.set_adaptive_sight(adaptive_sight)

    def step(self, actions):
        """把 info 留下來回傳"""
        rew, done, info = super().step(actions)
        info.update(self._info)
        return rew, done, info


class _MetaDriveGymmaWrapper(_MyGymmaWrapper):

    def step(self, actions):
        """MetaDrive 的 info 東西太多，不回傳回去，不然log會有問題。
        MetaDrive 要傳東西的話，可以改寫 get_stats 方法。
        """
        rew, done, info = super().step(actions)
        return rew, done, {}


class _Resco2GymmaWrapper(_MyGymmaWrapper):
    pass


REGISTRY["gymma"] = partial(env_fn, env=_GymmaWrapper)
REGISTRY["metadrive"] = partial(env_fn, env=_MetaDriveGymmaWrapper)
REGISTRY["resco2"] = partial(env_fn, env=_Resco2GymmaWrapper)


# -----------------------------------------------------------------------
#cityflow的環境設置
'''
class CityFlowMultiAgentEnv(MultiAgentEnv):
    def __init__(self, key, time_limit, pretrained_wrapper, seed, add_sight_id_len: Optional[int], episode_limit=3600, default_sight=1, **kwargs):
        if key is None:
            raise ValueError("CityFlow config key cannot be None")

        cityflow_config = os.path.abspath(key)
        self.eng = Engine(cityflow_config, thread_num=1)

        # 讀 config.json
        with open(cityflow_config) as f:
            config_data = json.load(f)
        roadnet_path = os.path.join(config_data["dir"], config_data["roadnetFile"])
        with open(roadnet_path) as f:
            self.roadnet = json.load(f)

        # 解析 traffic light IDs
        self.traffic_light_ids = [
            i["id"] for i in self.roadnet["intersections"]
            if not i.get("virtual", False) and i.get("trafficLight", False)
        ]
        if not self.traffic_light_ids:
            raise RuntimeError("No traffic lights found in roadnet")

        self.n_agents = len(self.traffic_light_ids)
        self.episode_limit = episode_limit
        self.t = 0

        # 建立 traffic light 到其控制車道的映射
        self.tl_to_lanes = self._build_tl_to_lanes()

        # 視野與鄰接矩陣
        self.DEFAULT_SIGHT = default_sight
        self.cur_adaptive_sight = self.DEFAULT_SIGHT
        self.cityflow_adjacency = self._build_adjacency_matrix()

        self.max_obs_len = self._compute_max_obs_len()

        # 自己維護的 phase 狀態 (初始假設 0)
        self.current_phase = {tl_id: 0 for tl_id in self.traffic_light_ids}

    def _build_tl_to_lanes(self):
        tl_to_lanes = {tl: [] for tl in self.traffic_light_ids}
        for road in self.roadnet["roads"]:
            end_int = road["endIntersection"]
            if end_int in tl_to_lanes:
                for i in range(len(road["lanes"])):
                    lane_id = f"{road['id']}_{i}"
                    tl_to_lanes[end_int].append(lane_id)
        return tl_to_lanes

    def _build_adjacency_matrix(self):
        n = self.n_agents
        adj = np.zeros((n, n), dtype=np.int32)
        id2idx = {tl: idx for idx, tl in enumerate(self.traffic_light_ids)}
        for road in self.roadnet["roads"]:
            s = road["startIntersection"]
            e = road["endIntersection"]
            if s in id2idx and e in id2idx:
                i, j = id2idx[s], id2idx[e]
                adj[i, j] = 1
                adj[j, i] = 1
        return adj
    
    def _compute_max_obs_len(self):
        """計算所有 agent 最大的 obs 長度，用來 padding"""
        max_len = 0
        for i in range(len(self.traffic_light_ids)):
            neighbors = self.get_neighbors_within_sight(i)
            cur_len = 2 + len(neighbors) * 2  # 自己2 + 每個鄰居2
            max_len = max(max_len, cur_len)
        return max_len

    def get_neighbors_within_sight(self, agent_id, sight=None):
        if sight is None:
            sight = self.cur_adaptive_sight
        visited = set()
        frontier = [agent_id]
        for _ in range(sight):
            next_frontier = []
            for u in frontier:
                for v, conn in enumerate(self.cityflow_adjacency[u]):
                    if conn and v not in visited:
                        next_frontier.append(v)
                        visited.add(v)
            frontier = next_frontier
        visited.discard(agent_id)
        return list(visited)

    def reset(self):
        self.eng.reset()
        self.t = 0
        # reset phase states to 0
        for tl in self.current_phase:
            self.current_phase[tl] = 0
        return self.get_obs(), self.get_state()

    def step(self, actions: List[int]):
        for i, action in enumerate(actions):
            tl_id = self.traffic_light_ids[i]
            self.eng.set_tl_phase(tl_id, action)
            self.current_phase[tl_id] = action  # 更新自己的相位紀錄

        self.eng.next_step()
        self.t += 1

        obs = self.get_obs()
        state = self.get_state()
        reward = self.calculate_reward()
        terminated = (self.t >= self.episode_limit)
        info = {"throughput": self.eng.get_throughput(),
                "delay": self.eng.get_delay()}
        return reward, terminated, info

    def get_obs_agent(self, agent_id):
        tl_id = self.traffic_light_ids[agent_id]
        obs = []

        lane_waiting = self.eng.get_lane_waiting_vehicle_count()
        waiting_count = sum(lane_waiting.get(l, 0) for l in self.tl_to_lanes[tl_id])

        obs.append(float(self.current_phase[tl_id]))
        obs.append(float(waiting_count))

        neighbors = self.get_neighbors_within_sight(agent_id)
        for nb in neighbors:
            nb_tl = self.traffic_light_ids[nb]
            nb_wait = sum(lane_waiting.get(l, 0) for l in self.tl_to_lanes[nb_tl])
            obs.append(float(self.current_phase[nb_tl]))
            obs.append(float(nb_wait))

        # Padding
        if len(obs) < self.max_obs_len:
            obs += [0.0] * (self.max_obs_len - len(obs))

        return np.array(obs, dtype=np.float32)

    def get_obs(self):
        """回傳所有 agent 的固定長度觀測"""
        return [self.get_obs_agent(i) for i in range(self.n_agents)]

    def get_obs_size(self):
        """提供給模型 input dim"""
        return self.max_obs_len

    def get_state(self):
        """拼接所有 agent 的觀測作為全域 state"""
        return np.concatenate(self.get_obs(), axis=0).astype(np.float32)
    
    def get_state_size(self):
        """Returns the total state dimension for CityFlow"""
        return self.n_agents * self.max_obs_len

    def get_avail_agent_actions(self, agent_id):
        tl_id = self.traffic_light_ids[agent_id]
        phase_count = 9  # 若你知道每個 intersection 有幾相，可改成自動讀
        return [1] * phase_count

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_total_actions(self):
        return len(self.get_avail_agent_actions(0))


    def calculate_reward(self):
        lane_waiting = self.eng.get_lane_waiting_vehicle_count()
        total_wait = 0
        for tl in self.traffic_light_ids:
            total_wait += sum(lane_waiting.get(l, 0) for l in self.tl_to_lanes[tl])
        return -float(total_wait)

    def set_adaptive_sight(self, sight):
        self.cur_adaptive_sight = sight

    def render(self):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        import numpy as _np
        _np.random.seed(seed)

    def save_replay(self):
        pass

    def get_stats(self):
        # 暫時不回傳任何東西
        return {}
    

    def get_env_info(self):
        return {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
            "cityflow_adjacency": self.cityflow_adjacency,
        }



REGISTRY["cityflow"] = partial(env_fn, env=CityFlowMultiAgentEnv)
'''
class CityFlowMultiAgentEnv(MultiAgentEnv):
    def __init__(
        self, key, time_limit, pretrained_wrapper, seed,
        add_sight_id_len: Optional[int],
        episode_limit=3600,
        default_sight=1,
        max_sight=3,          # 新增：最大 sight
        sight_mode='hop',  # 'grid' (n×n方格) 或 'cross' (十字) 或 'hop' (原始)
        **kwargs
    ):
        # ---------- 原本初始化（略，未動） ----------
        cityflow_config = os.path.abspath(key)
        self.eng = Engine(cityflow_config, thread_num=1)

        with open(cityflow_config) as f:
            config_data = json.load(f)
        roadnet_path = os.path.join(config_data["dir"], config_data["roadnetFile"])
        with open(roadnet_path) as f:
            self.roadnet = json.load(f)

        self.road_lengths = self.load_road_lengths(roadnet_path)

        self.traffic_light_ids = [
            i["id"] for i in self.roadnet["intersections"]
            if not i.get("virtual", False) and i.get("trafficLight", False)
        ]

        self.n_agents = len(self.traffic_light_ids)
        self.episode_limit = episode_limit
        self.t = 0

        self.tl_to_lanes = self._build_tl_to_lanes()

        # ---------- sight 設定 ----------
        self.MAX_SIGHT = max_sight
        self.cur_adaptive_sight = default_sight
        self.sight_mode = sight_mode  # 'grid', 'cross', or 'hop'
        self.sight_buckets = [25,50,75,100]

        # 提取網格座標（從 intersection_col_row 格式）
        self.tl_grid_coords = self._extract_grid_coordinates()

        self.cityflow_adjacency = self._build_adjacency_matrix()

        # print(f"traffic lights ids = {self.traffic_light_ids}")
        # print(self.tl_to_lanes)
        # print(self.cityflow_adjacency)

        # obs 長度永遠用 MAX_SIGHT 算
        self.max_obs_len = self._compute_max_obs_len()

        self.last_total_wait = 0.0

        self.current_phase = {tl_id: 0 for tl_id in self.traffic_light_ids}

        self.episode = 0
        self.tl_waiting_history = {}
        self.dsr_intersections = []
        self.fixed_distance = [30,30,30]

    def _extract_grid_coordinates(self):
        """
        從 intersection_X_Y 格式的 ID 提取網格座標
        例如: intersection_1_5 → (col=1, row=5)
        
        注意：只處理實體路口（已在 traffic_light_ids 中過濾掉 virtual 路口）
        """
        coords = {}
        for i, tl_id in enumerate(self.traffic_light_ids):
            try:
                # 分割 ID: "intersection_1_5" → ["intersection", "1", "5"]
                parts = tl_id.split('_')
                if len(parts) >= 3:
                    col = int(parts[1])  # X 座標（列）
                    row = int(parts[2])  # Y 座標（行）
                    coords[i] = (col, row)
                else:
                    raise ValueError(f"無法解析路口 ID: {tl_id}")
            except (ValueError, IndexError) as e:
                print(f"警告: 無法解析路口 ID {tl_id}, 錯誤: {e}")
                # 如果解析失敗，使用索引作為座標
                coords[i] = (i % 10, i // 10)
        
        # 打印路口座標資訊（除錯用）
        if len(coords) > 0:
            cols = [c for c, r in coords.values()]
            rows = [r for c, r in coords.values()]
            print(f"[INFO] 實體路口數量: {len(coords)}")
            print(f"[INFO] 列範圍: {min(cols)} ~ {max(cols)}")
            print(f"[INFO] 行範圍: {min(rows)} ~ {max(rows)}")
        
        return coords
    
    # def get_neighbors_within_sight(self, agent_id, sight):
    #     """
    #     回傳 List[(neighbor_id, hop)]
    #     """
    #     visited = set([agent_id])
    #     frontier = [(agent_id, 0)]
    #     neighbors = []

    #     while frontier:
    #         u, dist = frontier.pop(0)
    #         if dist == sight:
    #             continue

    #         for v, conn in enumerate(self.cityflow_adjacency[u]):
    #             if conn and v not in visited:
    #                 visited.add(v)
    #                 neighbors.append((v, dist + 1))
    #                 frontier.append((v, dist + 1))

    #     return neighbors
    
    def get_neighbors_within_sight(self, agent_id, sight):
        """
        根據 sight_mode 返回不同範圍的鄰居
        Returns: List[(neighbor_id, distance)]
        """
        if self.sight_mode == 'grid':
            return self._get_grid_neighbors(agent_id, sight)
        elif self.sight_mode == 'cross':
            return self._get_cross_neighbors(agent_id, sight)
        else:  # 'hop'
            return self._get_hop_neighbors(agent_id, sight)
    
    def _get_grid_neighbors(self, agent_id, sight):
        """
        n×n 方格視野（包含對角線）
        sight=1 → 3×3 = 8個鄰居
        sight=2 → 5×5 = 24個鄰居
        
        使用切比雪夫距離: max(|Δcol|, |Δrow|) <= sight
        Returns: List[(neighbor_id, chebyshev_distance)]
        """
        my_col, my_row = self.tl_grid_coords[agent_id]
        neighbors = []
        
        for nb_id, (nb_col, nb_row) in self.tl_grid_coords.items():
            if nb_id == agent_id:
                continue
            
            col_diff = abs(nb_col - my_col)
            row_diff = abs(nb_row - my_row)
            
            # 切比雪夫距離（方格距離）
            chebyshev_dist = max(col_diff, row_diff)
            
            if chebyshev_dist <= sight:
                neighbors.append((nb_id, chebyshev_dist))
        
        neighbors.sort(key=lambda x: (x[1], x[0]))  # 先按距離排序，再按 ID

        # print(agent_id)
        # print(sight)
        # print(f"current sight = {self.cur_adaptive_sight}")
        # print(neighbors)

        return neighbors
    
    def _get_cross_neighbors(self, agent_id, sight):
        """
        十字視野（只看上下左右，不包含對角線）
        sight=1 → 4個鄰居（上下左右各1個）
        sight=2 → 8個鄰居（上下左右各2個）
        
        只選擇 col 相同或 row 相同的鄰居
        Returns: List[(neighbor_id, manhattan_distance)]
        """
        my_col, my_row = self.tl_grid_coords[agent_id]
        neighbors = []
        
        for nb_id, (nb_col, nb_row) in self.tl_grid_coords.items():
            if nb_id == agent_id:
                continue
            
            # 只選擇同一行或同一列的鄰居
            if nb_col == my_col or nb_row == my_row:
                col_diff = abs(nb_col - my_col)
                row_diff = abs(nb_row - my_row)
                manhattan_dist = col_diff + row_diff
                
                if manhattan_dist <= sight:
                    neighbors.append((nb_id, manhattan_dist))
        
        neighbors.sort(key=lambda x: (x[1], x[0]))

        # print(agent_id)
        # print(sight)
        # print(f"current sight = {self.cur_adaptive_sight}")
        # print(neighbors)

        return neighbors
    
    def _get_hop_neighbors(self, agent_id, sight):
        """
        原始的 hop-based 鄰居搜尋（圖距離）
        Returns: List[(neighbor_id, hop)]
        """
        visited = set([agent_id])
        frontier = [(agent_id, 0)]
        neighbors = []

        while frontier:
            u, dist = frontier.pop(0)
            if dist == sight:
                continue

            for v, conn in enumerate(self.cityflow_adjacency[u]):
                if conn and v not in visited:
                    visited.add(v)
                    neighbors.append((v, dist + 1))
                    frontier.append((v, dist + 1))

        # print(agent_id)
        # print(sight)
        # print(f"current sight = {self.cur_adaptive_sight}")
        # print(neighbors)

        return neighbors

    def _build_tl_to_lanes(self):
        tl_to_lanes = {tl: [] for tl in self.traffic_light_ids}
        for road in self.roadnet["roads"]:
            end_int = road["endIntersection"]
            if end_int in tl_to_lanes:
                for i in range(len(road["lanes"])):
                    lane_id = f"{road['id']}_{i}"
                    tl_to_lanes[end_int].append(lane_id)
        return tl_to_lanes

    def _build_adjacency_matrix(self):
        n = self.n_agents
        adj = np.zeros((n, n), dtype=np.int32)
        id2idx = {tl: idx for idx, tl in enumerate(self.traffic_light_ids)}
        for road in self.roadnet["roads"]:
            s = road["startIntersection"]
            e = road["endIntersection"]
            if s in id2idx and e in id2idx:
                i, j = id2idx[s], id2idx[e]
                adj[i, j] = 1
                adj[j, i] = 1
        return adj
    '''
    def _compute_max_obs_len(self):
        """用 MAX_SIGHT 計算 obs 最大長度 每個node3維"""
        max_len = 0
        for i in range(self.n_agents):
            neighbors = self.get_neighbors_within_sight(i, sight=self.MAX_SIGHT)
            cur_len = 3 + len(neighbors) * 3  # phase + wait + vehicle_count + avg_speed + hop
            max_len = max(max_len, cur_len)
        return max_len
    '''
    # def _compute_max_obs_len(self):
    #     """用 MAX_SIGHT 計算 obs 最大長度 每個node3維"""
    #     max_len = 0
    #     for tl_id in self.traffic_light_ids:
    #         length = 0
    #         for lane_id in self.tl_to_lanes[tl_id]:
    #             length += 1
    #         if length> max_len:
    #              max_len = length
    #     return max_len + 1  # 加上 phase 的維度
    
    def _compute_max_obs_len(self):
        """
        計算 obs 最大長度：
        - 每個路口 phase 佔 1 維
        - 每條 lane 的每個 bucket 各佔 1 維
        """
        max_len = 0
        for tl_id in self.traffic_light_ids:
            num_lanes = len(self.tl_to_lanes[tl_id])
            length = num_lanes * len(self.sight_buckets)
            if length > max_len:
                max_len = length

        return max_len + 1  # 加上 phase 的維度
    
    def reset(self):
        self.eng.reset()
        self.t = 0
        
        initial_local_waits, initial_total_wait = self.calculate_reward()
        self.last_local_waits = initial_local_waits
        self.last_total_wait = initial_total_wait

        self.episode += 1
        # 只有第一個 episode 才清空 history
        if self.episode == 1:
            self.tl_waiting_history = {}

        # reset phase states to 0
        for tl in self.current_phase:
            self.current_phase[tl] = 0
        return self.get_obs(), self.get_state()

    def step(self, actions: List[int]):
        for i, action in enumerate(actions):
            tl_id = self.traffic_light_ids[i]
            self.eng.set_tl_phase(tl_id, action)
            self.current_phase[tl_id] = action  # 更新自己的相位紀錄

        # Switch or not
        # for i, action in enumerate(actions):
        #     tl_id = self.traffic_light_ids[i]
        #     if action == 1:
        #         self.eng.set_tl_phase(tl_id, (self.current_phase[tl_id] + 1) % 9)
        #         self.current_phase[tl_id] = (self.current_phase[tl_id] + 1) % 9  # 更新自己的相位紀錄

        # 一次執行15秒
        for _ in range(5):
            self.eng.next_step()
        self.t += 1

        # if self.episode == 1:
        #     lane_waiting = self.eng.get_lane_waiting_vehicle_count()

        #     for tl_id, lanes in self.tl_to_lanes.items():
        #         total_q = 0
        #         for lane in lanes:
        #             total_q += lane_waiting.get(lane, 0)

        #         if tl_id not in self.tl_waiting_history:
        #             self.tl_waiting_history[tl_id] = []

        #         self.tl_waiting_history[tl_id].append(total_q)

        obs = self.get_obs()
        state = self.get_state()
        
        current_local_waits, current_total_sum = self.calculate_reward()
        #print(f'[ CityFlow ] Step {self.t}, Reward: {reward}')
        local_rewards = [
                    cur - last for cur, last in zip(current_local_waits, self.last_local_waits)
                ]
        
        # 全局 Reward (用於神經網路訓練)
        global_reward = sum(local_rewards)
        
        # 更新紀錄
        self.last_local_waits = current_local_waits
        self.last_total_wait = current_total_sum

        #reward = self.calculate_weighted_reward()
        #print(f'[ CityFlow ] Step {self.t}, Reward: {reward}')

        terminated = (self.t >= self.episode_limit)

        # if terminated and self.episode == 1:
        #     import numpy as np

        #     print("\n===== Selecting DSR Intersections =====")

        #     self.dsr_intersections = []

        #     for tl_id, qs in self.tl_waiting_history.items():
        #         max_q = float(np.max(qs))

        #         if max_q > 20:
        #             self.dsr_intersections.append(tl_id)
        #             print(f"DSR Enabled: {tl_id} (max={max_q:.2f})")

        #     print("================================\n")
        
        # lane_waiting = self.eng.get_lane_waiting_vehicle_count()
        # lane_total = self.eng.get_lane_vehicle_count()
        # print(obs)

        # lane_id = "road_1_0_1_1"

        # lane_vehicles = self.eng.get_lane_vehicles()
        # vehicle_speed = self.eng.get_vehicle_speed()
        # vehicle_distance = self.eng.get_vehicle_distance()

        # print(f"\n=== Lane Detail: {lane_id} ===")

        # vehicles = lane_vehicles.get(lane_id, [])

        # # 按距離排序（超重要）
        # vehicles_sorted = sorted(vehicles, key=lambda vid: vehicle_distance[vid])

        # for vid in vehicles_sorted:
        #     speed = vehicle_speed[vid]
        #     dist = vehicle_distance[vid]
        #     is_waiting = speed < 0.1

        #     print(f"id={vid}, dist={dist:.1f}, speed={speed:.2f}, waiting={is_waiting}")

        # print("================================\n")

        
        info = {"throughput": self.eng.get_throughput(),
                "delay": self.eng.get_delay(),
                "average_travel_time": self.eng.get_average_travel_time(),
                "local_rewards": local_rewards}
        
        return global_reward, terminated, info
    
    def get_avg_speed_of_intersection(self, tl_id):
        speeds = []
        for lane in self.tl_to_lanes[tl_id]:
            for vid in self._lane_vehicles.get(lane, []):
                if vid in self._vehicle_speeds:
                    speeds.append(self._vehicle_speeds[vid])

        if not speeds:
            return 0.0
        return float(np.mean(speeds))
    
    def get_vehicle_count_of_intersection(self, tl_id):
        return sum(
            self._lane_vehicle_cnt.get(l, 0)
            for l in self.tl_to_lanes[tl_id]
        )
    '''
    # 根據 agent_id 和目前的視野設定，回傳該 agent 的觀測值（包含鄰居資訊）
    def get_obs_agent(self, agent_id):
        tl_id = self.traffic_light_ids[agent_id]
        obs = []

        #lane_waiting = self.eng.get_lane_waiting_vehicle_count()
        waiting_count = sum(
            self._lane_waiting.get(l, 0) for l in self.tl_to_lanes[tl_id]
        )
        #avg_speed = self.get_avg_speed_of_intersection(tl_id)
        #vehicle_count = self.get_vehicle_count_of_intersection(tl_id)

        # self (hop = 0)
        obs.append(float(self.current_phase[tl_id]))
        obs.append(float(waiting_count))
        #obs.append(float(vehicle_count))
        #obs.append(avg_speed)
        obs.append(0.0)

        neighbors = self.get_neighbors_within_sight(
            agent_id, sight=self.MAX_SIGHT
        )

        for nb, hop in neighbors:
            nb_tl = self.traffic_light_ids[nb]
            nb_wait = sum(
                self._lane_waiting.get(l, 0) for l in self.tl_to_lanes[nb_tl]
            )
            #nb_avg_speed = self.get_avg_speed_of_intersection(nb_tl)
            #nb_vehicle_count = self.get_vehicle_count_of_intersection(nb_tl)

            obs.append(float(self.current_phase[nb_tl]))
            obs.append(float(nb_wait))
            #obs.append(float(nb_vehicle_count))
            #obs.append(nb_avg_speed)
            obs.append(float(hop))  # hop 資訊

        if len(obs) < self.max_obs_len:
            obs += [0.0] * (self.max_obs_len - len(obs))

        return np.array(obs, dtype=np.float32)
    '''
    # 根據 agent_id 回傳該 agent 的觀測值與道路長度
    # def get_obs_agent(self, agent_id):
    #     tl_id = self.traffic_light_ids[agent_id]
    #     obs = []

    #     obs.append(float(self.current_phase[tl_id]))

    #     for lane_id in self.tl_to_lanes[tl_id]:
    #         queue = self._lane_waiting.get(lane_id, 0)

    #         if tl_id in self.dsr_intersections:
    #             if queue > 5:
    #                 obs.append(float(5))
    #             else:
    #                 obs.append(float(queue))
    #         else:
    #             obs.append(float(queue))
    #         '''
    #         # 從 lane_id 提取 road_id
    #         # 'road_5_3_2_0' -> 'road_5_3_2'
    #         road_id = '_'.join(lane_id.split('_')[:-1])
            
    #         # 取得道路長度
    #         road_length = self.road_lengths.get(road_id, 1.0)
    #         obs.append(self.road_lengths[tl_id])
    #         '''

    #     if len(obs) < self.max_obs_len:
    #         obs += [0.0] * (self.max_obs_len - len(obs))

    #     return np.array(obs, dtype=np.float32)
    
    #根據距離
    def get_obs_agent(self, agent_id):
        tl_id = self.traffic_light_ids[agent_id]
        obs = []

        # phase
        obs.append(float(self.current_phase[tl_id]))

        vehicle_distance = self.eng.get_vehicle_distance()

        # # 🔥 判斷是否為 DSR 路口
        # if tl_id in self.dsr_intersections:
        #     sight_buckets = self.sight_buckets   # 多距離
        # else:
        #     sight_buckets = self.fixed_distance  # 固定距離

        for lane_id in self.tl_to_lanes[tl_id]:

            vehicles = self._lane_vehicles.get(lane_id, [])

            # lane 長度
            road_id = "_".join(lane_id.split("_")[:-1])
            lane_length = self.road_lengths.get(road_id, 200)

            prev_dist = 0

            for bucket_dist in self.sight_buckets:

                count_in_bucket = sum(
                    1
                    for vid in vehicles
                    if  prev_dist < (lane_length - vehicle_distance.get(vid, 0)) <= bucket_dist
                )

                obs.append(float(count_in_bucket))

                prev_dist = bucket_dist

        # padding
        if len(obs) < self.max_obs_len:
            obs += [0.0] * (self.max_obs_len - len(obs))

        return np.array(obs, dtype=np.float32)

    
    def get_obs(self):
        self._lane_waiting = self.eng.get_lane_waiting_vehicle_count()
        #self._lane_vehicle_cnt = self.eng.get_lane_vehicle_count()
        self._lane_vehicles = self.eng.get_lane_vehicles()
        #self._vehicle_speeds = self.eng.get_vehicle_speed()
        return [self.get_obs_agent(i) for i in range(self.n_agents)]

    def get_obs_size(self):
        """提供給模型 input dim (preprocess 後)"""
        
        max_lanes = 0
        for tl_id in self.traffic_light_ids:
            num_lanes = len(self.tl_to_lanes[tl_id])
            if num_lanes > max_lanes:
                max_lanes = num_lanes

        return max_lanes + 1

    def get_state(self):
        """拼接所有 agent 的觀測作為全域 state"""
        return np.concatenate(self.get_obs(), axis=0).astype(np.float32)
    
    def get_state_size(self):
        """Returns the total state dimension for CityFlow"""
        max_lanes = 0
        for tl_id in self.traffic_light_ids:
            num_lanes = len(self.tl_to_lanes[tl_id])
            if num_lanes > max_lanes:
                max_lanes = num_lanes
        return self.n_agents * (max_lanes + 1) #(self.max_obs_len // 3 * 2)  # 不包含 hop 資訊

    def get_avail_agent_actions(self, agent_id):
        tl_id = self.traffic_light_ids[agent_id]
        phase_count = 9  # 若你知道每個 intersection 有幾相，可改成自動讀
        return [1] * phase_count
    
        # # Switch or not
        # action 0: keep
        # action 1: switch
        # return [1, 1]

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_total_actions(self):
        return len(self.get_avail_agent_actions(0))

    def load_road_lengths(self, config_file):
        """從 roadnet JSON 讀取所有道路長度"""
        with open(config_file, 'r') as f:
            roadnet = json.load(f)
        
        road_lengths = {}
        for road in roadnet['roads']:
            road_id = road['id']
            points = road['points']
            
            # 計算起點到終點的距離
            x1, y1 = points[0]['x'], points[0]['y']
            x2, y2 = points[-1]['x'], points[-1]['y']
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            road_lengths[road_id] = length if length > 0 else 1.0
        
        return road_lengths
    
    def calculate_reward(self):
        lane_waiting = self.eng.get_lane_waiting_vehicle_count()
        local_rewards = []  # 用來存每個路口的 reward
        total_wait = 0

        for tl in self.traffic_light_ids:
            tl_wait = 0  # 該路口的等待數
            for l in self.tl_to_lanes[tl]:
                w = lane_waiting.get(l, 0)
                tl_wait += w
            
            # CityFlow 中等待數愈多 reward 愈低，所以取負值
            local_rewards.append(-float(tl_wait))
            total_wait += tl_wait

        # 回傳兩個值：局部獎勵清單，以及總獎勵
        return local_rewards, -float(total_wait)
    
    #計算所有車輛數
    # def calculate_reward(self):
    #     total_vehicle = 0

    #     for tl in self.traffic_light_ids:
    #         for lane_id in self.tl_to_lanes[tl]:

    #             vehicles = self._lane_vehicles.get(lane_id, [])

    #             # 只算可觀察範圍內
    #             total_vehicle += len(vehicles)

    #     return -float(total_vehicle)
    
    def calculate_weighted_reward(self):
        lane_waiting = self.eng.get_lane_waiting_vehicle_count()
        total_wait = 0
        for tl in self.traffic_light_ids:
            for lane_id in self.tl_to_lanes[tl]:
                waiting_count = lane_waiting.get(lane_id, 0)
                
                # 從 lane_id 提取 road_id
                # 'road_5_3_2_0' -> 'road_5_3_2'
                road_id = '_'.join(lane_id.split('_')[:-1])
                
                # 取得道路長度
                road_length = self.road_lengths.get(road_id, 1.0)

                #print(road_length)
                #print(waiting_count)
                # 加權計算
                total_wait += waiting_count / (road_length)

                #print(total_wait)
        
        return -float(total_wait)

    def set_adaptive_sight(self, sight):
        self.cur_adaptive_sight = sight

    def render(self):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        import numpy as _np
        _np.random.seed(seed)

    def save_replay(self):
        pass

    def get_stats(self):
        # 暫時不回傳任何東西
        return {}
    

    def get_env_info(self):
        return {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
            "cityflow_adjacency": self.cityflow_adjacency,
        }



REGISTRY["cityflow"] = partial(env_fn, env=CityFlowMultiAgentEnv)

