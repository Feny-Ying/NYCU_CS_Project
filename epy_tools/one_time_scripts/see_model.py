import os
import torch as th

ROOT_PATH='/media/ppo/workspace/epymarl_research/result_1104_lbf_UCB_Ablation_longer1M/models/qmix DSR ucb(1s,2s,3s,4s,5s,6s,7s,8s,9s,10s)_seed1_lbforaging:Foraging-10s-10x10-4p-4f-coop-v1_2024-11-06 00:47:09.998752/200050'

AGENT_PATH = os.path.join(ROOT_PATH, 'agent.th')
MIXER_PATH = os.path.join(ROOT_PATH, 'mixer.th')
OPT_PATH = os.path.join(ROOT_PATH, 'opt.th')

# Load all of them
AGENT = th.load(AGENT_PATH)
MIXER = th.load(MIXER_PATH)
OPT = th.load(OPT_PATH)

# See the structure
print(AGENT.keys())
print(MIXER.keys())
print(OPT.keys())

# See the agent structure and the width of all layers
for k, v in AGENT.items():
    print(k, v.shape)