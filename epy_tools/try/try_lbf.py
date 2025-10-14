import gym

env = gym.make('lbforaging:SEForaging-15s-1_2_3_4_5_6_15-15x15-3p-3f-v1')

env = gym.make('lbforaging:Foraging-2s-15x15-3p-3f-v1')

# env = gym.make('lbforaging:Foraging-2s-5x5-2p-2f-v1')
# env.seed(0)
# obs = env.reset()
# env.render()
#
# print(obs)


def get_start_obs(env_, seed=0):
    print('--------------------------------------')
    from epy_tools.lbf_utils import pprint_obs
    env_.seed(seed)
    obs_ = env_.reset()
    env_.render()
    pprint_obs(obs_, env_.n_agents, env_.max_food)
    # return obs_

env.env.sight=5
get_start_obs(env, 0)
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
