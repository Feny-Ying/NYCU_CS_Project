import robotic_warehouse
import gym

# env = gym.make("rware-tiny-2ag-v1")
# env = gym.make("rware-small-2ag-v1")
# env = gym.make("rware-medium-2ag-v1")
env = gym.make("robotic_warehouse:rware-medium-10ag-v1")

a = env.reset()
env.render()

# noinspection PyUnresolvedReferences
import lbforaging
import gym

import resco_benchmark
env = gym.make('resco_benchmark:cologne3-1s-v1')

a = env.reset()



# from epy_tools.lbf_utils import turn_preprocess_schedule_str_to_func

# turn_preprocess_schedule_str_to_func('lbf:2s@0,4s@1e6,6s@2e6,8s@3e6,15@4e6', 2)(1500000)

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

