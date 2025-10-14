import gym
import numpy as np

# env = gym.make('lbforaging:CombinedForaging-15x15x2-2p-2f-coop-showAllSubEnv-v1')
# env.seed(0)
# obsShowAll = env.reset()

# env = gym.make('lbforaging:CombinedForaging-15x15x3-3p-1f-coop-showSelfSubEnv-v1')
# env = gym.make('lbforaging:CombinedForaging-10x10x3-2p-2f-showSelfSubEnv-v1')
# env = gym.make('lbforaging:CombinedForaging-10x10x2-2p-3f-coop-showSelfSubEnv-v1')


env = gym.make('lbforaging:CombinedForaging-10x10x2-2p-3f-showSelfSubEnv-fixBoard-v1')

# env = gym.make('lbforaging:CombinedForaging-10x10x5-2p-3f-showSelfSubEnv-v1')
# env = gym.make('lbforaging:CombinedForaging-15x15x2-3p-5f-showSelfSubEnv-v1')

# env = gym.make('lbforaging:CombinedForaging-15x15x2-2p-2f-coop-showSelfSubEnv-v1')
env.seed(0)
obsShowSelf = env.reset()
env.seed(0); env.reset(); env.render()
env.seed(1); env.reset(); env.render()
env.seed(2); env.reset(); env.render()

env.render()


def fff():
    return np.array([env.envs[0].action_space.sample() for _ in range(len(env.envs))]).flatten().tolist()
env.step(fff()); env.render()

raise Exception('stop here')




# print difference (regions that are different are highlighted by 1 otherwise 0)
diff = (obsShowAll != obsShowSelf).astype(int)

print(obsShowAll.reshape(3, 3, 36)[0])
print(obsShowSelf.reshape(3, 3, 36)[0])

print(diff.reshape(3, 3, 36))


print(obsShowAll)
print(obsShowSelf)



a = np.zeros((2,3))
a







raise Exception('stop here')


import robotic_warehouse
import gym

# env = gym.make('lbforaging:ForagingFI-1s-5x5-2p-2f-v1')
env = gym.make('lbforaging:ForagingFI-4s-15x15-3p-5f-v1')
# env = gym.make('lbforaging:ForagingFI-4s-15x15-5p-5f-v1')
# env.seed(0)
obs = env.reset()
env.render()
raise Exception('stop here')

print(obs)

# env = gym.make('lbforaging:Foraging-2s-15x15-4p-5f-coop-v1')
env = gym.make('lbforaging:Foraging-2s-15x15-5p-5f-v1')
# env = gym.make('lbforaging:Foraging-2s-5x5-2p-2f-v1')
env.seed(0)
obs = env.reset()
env.render()

print(obs)

raise Exception('stop here')
env.sight = 2
print(env.seed(0))

a = env.reset()
env.render()

raise Exception('stop here')

env.action_space

obs, *_ = env.step((1,1))
obs, *_ = env.step((0,2))
# env.step((2,2))
obs, *_ = env.step((3,3))
obs, *_ = env.step((4,4))
obs, *_ = env.step((2,0,2,2,0))
obs, *_ = env.step((0,0,0,0,3))
# env.step((0,0,2,2))
# env.step((5,0,5,5))
env.render()

# env.close()

