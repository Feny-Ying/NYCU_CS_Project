from metadrive import MultiAgentRoundaboutEnv, MultiAgentBottleneckEnv, MultiAgentIntersectionEnv, \
    MultiAgentParkingLotEnv, MultiAgentTollgateEnv

env_classes = [MultiAgentRoundaboutEnv,
               MultiAgentBottleneckEnv,
               MultiAgentIntersectionEnv,
               MultiAgentParkingLotEnv,
               MultiAgentTollgateEnv]

cfg = dict(
    # num_agents=20,
    num_agents=5,
           # use_render=True
)

# env = MultiAgentRoundaboutEnv(cfg)
# env = MultiAgentIntersectionEnv(cfg)
env = MultiAgentParkingLotEnv(cfg)
env.action_space

frames = []
print("Starting the environment {}\n".format(env))
env.reset()

tm = {"__all__": False}
for i in range(1000):
    if tm["__all__"]:
        frames.append(frame)
        continue
    action = env.action_space.sample()
    for a in action.values():
        a[-1] = 1.0
    o, r, tm, tc, info = env.step(action)
    frame = env.render(mode="top_down",
                       scaling=4,  # 4 pixels per meter
                       # camera_position=env.current_map.get_center_point(),
                       screen_size=(500, 500))
    frames.append(frame)
    print(f"Step {i}")
env.close()

from PIL import Image

imgs = [frame for frame in frames]
imgs = [Image.fromarray(img) for img in imgs]
imgs[0].save("demo.gif", save_all=True, append_images=imgs[1:], duration=50, loop=0)

raise ValueError("Stop here")
# frames = []
# for env_class in env_classes:
#     env = env_class()
#     print("Starting the environment {}\n".format(env))
#     env.reset()
#     tm={"__all__":False}
#     for i in range(100):
#         if tm["__all__"]:
#             frames.append(frame)
#             continue
#         action = env.action_space.sample()
#         for a in action.values():
#             a[-1] = 1.0
#         o,r,tm,tc,i = env.step(action)
#         frame = env.render(mode="top_down",
#                            scaling=4, # 4 pixels per meter
#                            camera_position=env.current_map.get_center_point(),
#                            screen_size=(500, 500))
#         frames.append(frame)
#     env.close()

# render image
print("\nGenerate gif...")
from PIL import Image

imgs = [frame for frame in frames]
imgs = [Image.fromarray(img) for img in imgs]
imgs[0].save("demo.gif", save_all=True, append_images=imgs[1:], duration=50, loop=0)
print("\nOpen gif...")
from IPython.display import Image

Image(open("demo.gif", 'rb').read())
