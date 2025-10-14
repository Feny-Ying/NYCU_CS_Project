import gym
import numpy as np
from gym.spaces import flatdim

import robotic_warehouse
from epy_tools.rware_utils import obs_parser_with_one_select_sight_info, rware_preprocess
from robotic_warehouse import Warehouse
from robotic_warehouse.warehouse import Action

np.random.seed(0)

env = gym.make(f'robotic_warehouse:rware-tiny-2ag-5s-v1')

env.seed(0)
OBS = env.reset()
env.render()

raise Exception('Stop here')

# obs_parser(sight=5, obs=OBS[0])

def step_an_action(env_, action_, parse_sight_):
    print('--------------------------------------')
    obs_, rew_, done_, _ = env_.step([action_] * env_.n_agents)
    env_.render()
    obs_parser_with_one_select_sight_info(sight=parse_sight_, obs=obs_[0])
    print(f'rew_={rew_}, done_={done_}')

step_an_action(env, action_=0, parse_sight_=SIGHT)




# rware_preprocess(new_sight=1, original_sight=SIGHT, original_obs_list=OBS, verbose=True)

raise Exception('Stop here')

AG1_OBS, AG2_OBS = OBS
obs_parser(AG1_OBS, sight=SIGHT)

obs_parser(AG2_OBS, sight=SIGHT)

print(env.step([Action.LEFT, Action.NOOP])[1:]);
env.render()
print(env.step([Action.RIGHT, Action.NOOP])[1:]);
env.render()
print(env.step([Action.FORWARD, Action.NOOP])[1:]);
env.render()
print(env.step([Action.TOGGLE_LOAD, Action.NOOP])[1:]);
env.render()

print(env.step([Action.NOOP, Action.LEFT])[1:]);
env.render()
print(env.step([Action.NOOP, Action.RIGHT])[1:]);
env.render()
print(env.step([Action.NOOP, Action.FORWARD])[1:]);
env.render()
print(env.step([Action.NOOP, Action.TOGGLE_LOAD])[1:]);
env.render()

print(env.step([Action.LEFT])[1:]);
env.render()
print(env.step([Action.RIGHT])[1:]);
env.render()
print(env.step([Action.FORWARD])[1:]);
env.render()
print(env.step([Action.TOGGLE_LOAD])[1:]);
env.render()

print(env.step([4, 4]));
env.render()

env = gym.make('robotic_warehouse:rware-tiny-2ag-easy-1s-v1')
env.reset()
env.render()

argv = dict(
    column_height=8,
    shelf_rows=1,
    shelf_columns=3,
    n_agents=2,
    msg_bits=0,
    sensor_range=3,
    request_queue_size=2,
    max_inactivity_steps=None,
    max_steps=500,
    reward_type=robotic_warehouse.RewardType.INDIVIDUAL,
)

np.random.seed(0)
envFast = Warehouse(fast_obs=True, **argv)
ag1F, ag2F = envFast.reset()
envFast.render()

np.random.seed(0)
envSlow = Warehouse(fast_obs=False, **argv)
ag1S, ag2S = envSlow.reset()
envSlow.render()

from pprint import pprint

print('-------------------- Fast ---------------------')
pprint(ag1F)
print('-------------------- Slow ---------------------')
pprint(ag1S)
print('-------------------- Slow + flatdim ---------------------')
pprint(flatdim(ag1S))

# import gym
#
# env = gym.make('robotic_warehouse:rware-small-2ag-easy-1s-v1')
#
# env._use_slow_obs()
#
# getattr(env, '_Warehouse_use_slow_obs', None)
#
# jjj = env.reset()
# env.render()
#
# ag1, ag2 = jjj
#
#
# def __getattr__(self, name):
#     # if name.startswith("_"):
#     #     raise AttributeError(
#     #         "attempted to get missing private attribute '{}'".format(name)
#     #     )
#     return getattr(self.env, name)
#
#
# def custom_getattr(self, name):
#     if name.startswith("_"):
#         raise AttributeError(
#             "attempted to get missing private attribute '{}'".format(name)
#         )
#     return getattr(self.env, name)
#
#
# env.__getattr__ = custom_getattr.__get__(env)
