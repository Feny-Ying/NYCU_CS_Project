import gym
from smac.env import MultiAgentEnv, StarCraft2Env

import pysc2.maps.melee

a = pysc2.maps.melee.Flat32()



env = StarCraft2Env()

obs = env.reset()

obs

SEED = 1

env = gym.make('lbforaging:ESMForaging-15s-1_2_3_4_5_6_15-15x15-4p-5f-v1')
env.seed(SEED)
from epy_tools.lbf_utils import pprint_se_obs


# env = gym.make('lbforaging:Foraging-2s-5x5-2p-2f-v1')
# env.seed(0)
# obs = env.reset()
# env.render()
#
# print(obs)


def get_start_obs(env_, seed=SEED):
    print('--------------------------------------')
    env_.seed(SEED)
    obs_ = env_.reset()
    env_.render()
    pprint_se_obs(obs_, env_.n_agents, env_.max_food)
    # return obs_


get_start_obs(env, SEED)


def step_an_action(env_, action_):
    print('--------------------------------------')
    obs_, rew_, done_, _ = env_.step([action_] * env_.n_agents)
    env_.render()
    pprint_se_obs(obs_, env_.n_agents, env_.max_food)
    print(f'rew_={rew_}, done_={done_}')

step_an_action(env, action_=0)
step_an_action(env, action_=1)





print(f'env.env.sight={env.env.sight}')
raise Exception('stop here')
print(env.seed(0))

a = env.reset()
env.render()

raise Exception('stop here')

env.action_space

obs, *_ = env.step((1, 1))
obs, *_ = env.step((0, 2))
# env.step((2,2))
obs, *_ = env.step((3, 3))
obs, *_ = env.step((4, 4))
obs, *_ = env.step((2, 0, 2, 2, 0))
obs, *_ = env.step((0, 0, 0, 0, 3))
# env.step((0,0,2,2))
# env.step((5,0,5,5))
env.render()

# env.close()
