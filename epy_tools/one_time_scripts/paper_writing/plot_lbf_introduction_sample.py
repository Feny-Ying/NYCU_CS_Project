import gym

# Make a 7-player environment
env = gym.make('lbforaging:Foraging-10s-10x10-5p-3f-v1')
obs = env.reset()
field = env.env.field
players = env.env.players

# Clear all fields and set the player position to the top-left corner
field[:, :] = 0
for i, player in enumerate(players):
    player.position = (i, 0)

# Player 0 (main)
players[0].position = (2, 7)
players[0].level = 1

# Fruit 0
field[1, 6] = 3

# Player 1 (near the main player) & fruit 1
players[1].position = (5, 6)
players[1].level = 1
field[4, 7] = 2

# Player 2 & 3 & fruit 2
players[2].position = (3, 1)
players[2].level = 2
players[3].position = (7, 2)
players[3].level = 1
field[5, 1] = 3

# Player 6 & fruit 4
players[4].position = (7, 8)
players[4].level = 1
field[9, 6] = 1

env.render()

raise NotImplementedError

# import gym
#
# # Make a 7-player environment
# env = gym.make('lbforaging:Foraging-15s-15x15-7p-5f-v1')
# obs = env.reset()
# field = env.env.field
# players = env.env.players
#
# # Clear all fields and set the player position to the top-left corner
# field[:, :] = 0
# for i, player in enumerate(players):
#     player.position = (i, 0)
#
# # Player 0 (main)
# players[0].position = (2, 12)
# players[0].level = 1
#
# # Fruit 0
# field[1, 11] = 3
#
# # Player 1 (near the main player) & fruit 1
# players[1].position = (6, 11)
# players[1].level = 1
# field[5, 12] = 2
#
# # Player 2 & 3 & fruit 2
# players[2].position = (2, 2)
# players[2].level = 2
# players[3].position = (5, 5)
# players[3].level = 1
# field[5, 3] = 3
#
# # Player 4 & 5 & fruit 3
# players[4].position = (9, 3)
# players[4].level = 1
# players[5].position = (12, 5)
# players[5].level = 1
# field[10, 6] = 3
#
# # Player 6 & fruit 4
# players[6].position = (10, 12)
# players[6].level = 1
# field[12, 11] = 1
#
# env.render()
#
# raise NotImplementedError

players[0].position = (0, 0)


field[0, 0] = 1
field

env.close()


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
