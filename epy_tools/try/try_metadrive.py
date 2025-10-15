# from IPython.display import Image
# from metadrive.utils import generate_gif
import os

from metadrive.envs.metadrive_env import MetaDriveEnv

sensor_size = (84, 60) if os.getenv('TEST_DOC') else (200, 100)

cfg = dict(
    # image_observation=True,
    # vehicle_config=dict(image_source="rgb_camera"),
    # sensors={"rgb_camera": (RGBCamera, *sensor_size)},
    # stack_size=3,
    # agent_policy=IDMPolicy  # drive with IDM policy
    use_render=True,
    vehicle_config=dict(lidar=dict(num_lasers=50, distance=50, num_others=0, gaussian_noise=0.0,
                                   dropout_prob=0.0, add_others_navi=False), )
)
env = MetaDriveEnv(cfg)
# env = MetaDriveEnv()
env.close()
frames = []

o, info = env.reset()

env.render(mode="top_down")
env.render(mode="human")

o, r, d, tr, info = env.step([0, 1]);
env.render()

print(o)

o
info

try:
    env.reset()
    for _ in range(1 if os.getenv('TEST_DOC') else 10000):
        # simulation
        o, r, d, _, _ = env.step([0, 1])
        # rendering, the last one is the current frame
        print(o)
        # ret=o["image"][..., -1]*255 # [0., 1.] to [0, 255]
        # ret=ret.astype(np.uint8)
        # frames.append(ret[..., ::-1])
        if d:
            break
    # generate_gif(frames if os.getenv('TEST_DOC') else frames[-300:-50])
finally:
    env.close()
