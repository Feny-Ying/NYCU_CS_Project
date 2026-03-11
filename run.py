import datetime
import os
import pprint
import threading
import time
from os.path import dirname, abspath
from types import SimpleNamespace as SN

import numpy as np
import torch as th

from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from controllers import REGISTRY as mac_REGISTRY
from epy_tools.preprocess import PreprocessManager
from learners import REGISTRY as le_REGISTRY
from modules.latent_models import REGISTRY as latent_model_REGISTRY
from runners import REGISTRY as r_REGISTRY
from utils.logging import Logger
from utils.timehelper import time_left, time_str


def run(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    # unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    try:
        map_name = _config["env_args"]["map_name"]
    except:
        map_name = _config["env_args"]["key"]
    # unique_token = f"{_config['name']}_seed{_config['seed']}_{map_name}_{datetime.datetime.now()}"
    unique_token = f"{_config['name']}_seed{_config['seed']}_{map_name}_{datetime.datetime.now()}"

    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(
            dirname(dirname(abspath(__file__))), args.save_path, "tb_logs"
        )
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    # os._exit(os.EX_OK)


def evaluate_sequential(args, runner, evaluate_save_buffer):
    if args.name in ['masia'] or 'masia' in args.name:  # MASIA 要求的
        print(args.test_nepisode)
        assert 0

    # if args.save_replay:
    #     # Save replay buffer
    #     from tqdm import tqdm
    #     for _ in tqdm(range(args.buffer_size), desc="Filling buffer", total=args.buffer_size):
    #         episode_batch = runner.run(test_mode=True)
    #         if evaluate_save_buffer is not None:
    #             evaluate_save_buffer.insert_episode_batch(episode_batch)
    #     buffer_dir = str(os.path.join(dirname(dirname(abspath(__file__))), args.save_path, "buffers"))
    #     os.makedirs(buffer_dir, exist_ok=True)
    #     buffer_path = str(os.path.join(buffer_dir, f"{args.unique_token}_buffer.pkl"))
    #     with open(buffer_path, 'wb') as f:
    #         pickle.dump(evaluate_save_buffer, f)
    #         print(f"Buffer saved to {buffer_path}")
    #     runner.close_env()
    #     return

    # Original
    for i_test in range(args.test_nepisode):
        print(f"Testing episode {i_test}")
        episode_batch = runner.run(test_mode=True)
        if evaluate_save_buffer is not None:
            evaluate_save_buffer.insert_episode_batch(episode_batch)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def get_buffer(scheme, groups, args, env_info, preprocess):
    return ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device
    )


def run_sequential(args, logger):
    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.cityflow_adjacency = env_info["cityflow_adjacency"]

    if 'add_sight_id_len' in args.env_args and args.env_args['add_sight_id_len'] is not None:
        sight_id_len_integer = int(args.env_args['add_sight_id_len'])
    else:
        sight_id_len_integer = None
    print(f"{args}")
    # Setup preprocess function
    preprocess_manager = PreprocessManager(
        n_agents=args.n_agents,
        preprocess_desc=args.preprocess_desc,
        args=args,
        reset_after_switch_visibility=args.reset_after_switch_visibility,
        reset_buffer_after_switch_visibility=args.reset_buffer_after_switch_visibility,
        add_sight_id_len=sight_id_len_integer
    )

    print(f'env_info: {env_info}')

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": th.int,
        },
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    # If preprocessing, add more schemes
    if preprocess_manager.adaptive_worker is not None:
        pass
    else:
        for possible_str in preprocess_manager.possible_visibility_str + [preprocess_manager.original_visibility_str]:
            scheme[f'obs_{possible_str}'] = {"vshape": env_info["obs_shape"], "group": "agents"}

    buffer = get_buffer(scheme, groups, args, env_info, preprocess)

    if args.save_replay:
        evaluate_save_buffer = get_buffer(scheme, groups, args, env_info, preprocess)
    else:
        evaluate_save_buffer = None

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args, preprocess_manager)

    # Latent model (for MASIA)
    if args.name in ['masia'] or 'masia' in args.name:
        latent_model = latent_model_REGISTRY[args.latent_model](args)
    else:
        latent_model = None

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    if args.name in ['masia'] or 'masia' in args.name:
        learner = le_REGISTRY[args.learner](mac, latent_model, buffer.scheme, logger, args)
    else:
        learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    # Setup preprocess function
    runner.preprocess_manager = preprocess_manager
    learner.preprocess_manager = preprocess_manager

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info(
                "Checkpoint directiory {} doesn't exist".format(args.checkpoint_path)
            )
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

    if args.evaluate or args.save_replay:
        assert args.test_encoder is False, '沒有加入 test_encoder 的功能，需要的話去看 MASIA 源代碼'
        runner.log_train_stats_t = runner.t_env
        evaluate_sequential(args, runner, evaluate_save_buffer)
        logger.log_stat("episode", runner.t_env, runner.t_env)
        logger.print_recent_stats()
        logger.console_logger.info("Finished Evaluation")
        return

    # 若有提供 buffer path，就 load 出來並用此 buffer 資料來 offline 訓練，之後再做一次 testing。
    if args.offline_buffer_path:
        runner.log_train_stats_t = runner.t_env
        import pickle
        with open(args.offline_buffer_path, 'rb') as f:
            loaded_buffer: ReplayBuffer = pickle.load(f)
        print(f'Offline buffer loaded from {args.offline_buffer_path}')
        print(loaded_buffer)
        assert loaded_buffer.can_sample(args.batch_size), 'Offline buffer cannot sample'
        from tqdm import tqdm

        # # Preprocess the whole buffer's obs
        cur_sight = preprocess_manager.try_get_adaptive_sight()
        new_batches = []
        for i_buffer in tqdm(range(loaded_buffer.buffer_size), desc="Preprocess buffer obs",
                             total=loaded_buffer.buffer_size):
            new_batch = runner.new_batch()
            # print(f'shape: {loaded_buffer[i_buffer]["obs"].shape}')
            ep_obs = loaded_buffer[i_buffer]["obs"].cpu().numpy()  # [1, ep_len+1, n_agents, obs_shape]
            # new_ep_obs = []
            for t_idx in range(ep_obs.shape[1]):
                ag_obs = ep_obs[0, t_idx]
                # print(f'cur_sight: {cur_sight}')
                transformed_obs = preprocess_manager.get_one_preprocessed_obs(ag_obs, visible_str=cur_sight)
                transformed_obs = th.tensor(transformed_obs, dtype=th.float32)
                # loaded_buffer[i_buffer].update({"obs": transformed_obs}, ts=t_idx)
                # print(f'available shape before: {loaded_buffer[i_buffer]["avail_actions"].shape}')
                avail_actions = loaded_buffer[i_buffer]["avail_actions"][0][t_idx]
                actions = loaded_buffer[i_buffer]["actions"][0][t_idx].view(1, -1)
                reward = loaded_buffer[i_buffer]["reward"][0][t_idx].view(1, -1)
                terminated = loaded_buffer[i_buffer]["terminated"][0][t_idx].view(1, -1)
                # print(
                #     f'ts: {t_idx}, avail_actions: {avail_actions}, actions: {actions}, reward: {reward}, terminated: {terminated}')
                new_batch.update(dict(obs=transformed_obs, avail_actions=avail_actions, actions=actions, reward=reward,
                                      terminated=terminated), ts=t_idx)
            new_batches.append(new_batch)
            #     # new_ep_obs.append(transformed_obs)
            # new_ep_obs = np.array(new_ep_obs)
            # new_ep_obs = new_ep_obs.reshape(1, *new_ep_obs.shape)
            # print(f'new_ep_obs.shape: {new_ep_obs.shape}')
            # new_ep_obs = th.tensor(new_ep_obs, dtype=th.float32)
            # # loaded_buffer[i_buffer]["obs"] = new_ep_obs
            # loaded_buffer[i_buffer].update({"obs": new_ep_obs})

        #
        print('Start inserting new batches')
        for new_batch in tqdm(new_batches, desc="Inserting new batches", total=len(new_batches)):
            buffer.insert_episode_batch(new_batch)

        # raise ValueError

        for _ in tqdm(range(args.offline_buffer_grad_steps), desc="Offline training",
                      total=args.offline_buffer_grad_steps):
            episode_sample = preprocess_manager.sample_current_needed_data(loaded_buffer, args, runner.t_env)

            # ep_obs = episode_sample['obs']  # shape: [batch_size, ep_len+1, n_agents, obs_shape]
            #
            # # Transform
            # for batch_idx in range(ep_obs.shape[0]):
            #     for t_idx in range(ep_obs.shape[1]):
            #         # Assuming using UCB 即使是 pure sight
            #         cur_sight = preprocess_manager.try_get_adaptive_sight()
            #         ag_obs = ep_obs[batch_idx, t_idx]  # shape: [n_agents, obs_shape]
            #         # To numpy
            #         ag_obs = ag_obs.cpu().numpy()
            #         transformed_obs = preprocess_manager.get_one_preprocessed_obs(ag_obs, visible_str=cur_sight)
            #
            #         # To tensor
            #         transformed_obs = th.tensor(transformed_obs, dtype=th.float32)
            #
            #         ep_obs[batch_idx, t_idx] = transformed_obs
            #
            #
            # episode_sample.update({'obs': ep_obs})

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]
            if episode_sample.device != args.device:
                episode_sample.to(args.device)
            learner.offline_train(episode_sample, runner.t_env, -1)  # The number is not used so arbitrary given -1
        print(f'Offline training finished')
        # Test
        for _ in range(args.test_nepisode):
            runner.run(test_mode=True)
        logger.log_stat("episode", runner.t_env, runner.t_env)
        logger.print_recent_stats()
        logger.console_logger.info("Finished Evaluation")
        return  # End the program

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    last_policy_diff1_T = 0
    last_policy_diff2_T = 0
    last_policy_diff3_T = 0
    last_reset_T = 0
    last_dormant_T = 0
    model_save_time = 0
    last_obs_key = None

    start_time = time.time()
    last_time = start_time
    n_reset_buffer = 0

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    if args.n_grad_steps > 1:
        for _ in range(10):
            print(f'Warning: args.n_grad_steps > 1, args.n_grad_steps: {args.n_grad_steps}')

    while runner.t_env <= args.t_max:

        # print(f"runner.t_env: {runner.t_env}")

        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode=False)

        buffer.insert_episode_batch(episode_batch)

        # Log Adaptive Worker
        if preprocess_manager.adaptive_worker is not None:
            if runner.t_env - learner.log_stats_t >= args.learner_log_interval:
                sight_to_times = preprocess_manager.adaptive_worker.get_each_sight_ratio_in_window()
                for sight, times in sight_to_times.items():
                    logger.log_stat(f"adaptive_worker/{sight}_times", times, runner.t_env)

        # Log Adaptive Worker
        if preprocess_manager.adaptive_worker is not None:
            if runner.t_env - learner.log_stats_t >= args.learner_log_interval:
                if preprocess_manager.adaptive_worker.n_arms > 1:
                    sight_to_exploitation = preprocess_manager.adaptive_worker.compute_each_sight_ucb_exploitation_value(t_env = runner.t_env)
                    sight_to_exploration = preprocess_manager.adaptive_worker.compute_each_sight_ucb_exploration_value()
                    sight_to_ucb_values = {sight: sight_to_exploitation[sight] + sight_to_exploration[sight] for sight
                                           in
                                           sight_to_exploitation}
                    # if inf exists, skip logging
                    if all([not np.isinf(v) for v in sight_to_ucb_values.values()]):
                        for sight in sight_to_ucb_values.keys():
                            if sight_to_exploitation is not None:
                                logger.log_stat(f"sight_exploit/{sight}", sight_to_exploitation[sight], runner.t_env)
                                logger.log_stat(f"sight_explore/{sight}", sight_to_exploration[sight], runner.t_env)
                                logger.log_stat(f"sight_ucb_total/{sight}", sight_to_ucb_values[sight], runner.t_env)
                history_len_enough = len(preprocess_manager.adaptive_worker.history) >= (
                        preprocess_manager.adaptive_worker.n_arms * args.batch_size_run)
                if preprocess_manager.adaptive_worker.n_arms == 1 or history_len_enough:
                    sight_to_times = preprocess_manager.adaptive_worker.get_each_sight_ratio_in_window()
                    if sight_to_times is not None:
                        for sight in sight_to_times.keys():
                            logger.log_stat(f"sight_times_in_window/{sight}", sight_to_times[sight], runner.t_env)

        # Assume no randomness in obs_key
        # If obs_key is switched, reset the buffer
        cur_obs_key = preprocess_manager.get_obs_key(t_env=runner.t_env)

        if args.reset_buffer_after_switch_visibility and last_obs_key is not None and last_obs_key != cur_obs_key:
            del buffer
            buffer = get_buffer(scheme, groups, args, env_info, preprocess)
            print(f"Resetting buffer; t_env: {runner.t_env}")
            n_reset_buffer += 1
            logger.log_stat("preprocess/n_reset_buffer", n_reset_buffer, runner.t_env)
        last_obs_key = cur_obs_key

        # ======= Training =======
        if buffer.can_sample(args.batch_size):

            for _ in range(args.n_grad_steps):
                episode_sample = preprocess_manager.sample_current_needed_data(buffer, args, runner.t_env)

                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != args.device:
                    episode_sample.to(args.device)

                learner.train(episode_sample, runner.t_env, assigned_obs_key=None)

                if args.train_next_visibility:
                    next_visibility_str = preprocess_manager.get_next_visibility_str(runner.t_env)
                    if next_visibility_str is not None:
                        next_obs_key = preprocess_manager.get_obs_key(t_env=None,
                                                                      visibility_str=next_visibility_str)
                        learner.train(episode_sample, runner.t_env, assigned_obs_key=next_obs_key)

            visibility_scalar = preprocess_manager.get_visibility_scalar_to_log(t_env=runner.t_env)
            if visibility_scalar is not None:
                if runner.t_env - learner.log_stats_t >= args.learner_log_interval:
                    logger.log_stat("preprocess/visibility_scalar", visibility_scalar, runner.t_env)
                    if args.train_next_visibility:
                        next_visibility_str = preprocess_manager.get_next_visibility_str(runner.t_env)
                        if next_visibility_str is not None:
                            next_visibility_scalar = preprocess_manager.get_visibility_scalar_to_log(
                                visibility_str=next_visibility_str)
                            logger.log_stat("preprocess/visibility_scalar_next", next_visibility_scalar, runner.t_env)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            #
            try:
                original_env = runner.env.original_env.env
                need_record_sight_history = args.runner == 'episode' and (
                        'SEForaging' in repr(original_env) or
                        'ESMForaging' in repr(original_env) or
                        'ESM' in repr(original_env)

                )
            except AttributeError:
                need_record_sight_history = False

            if need_record_sight_history:
                # --- 統計不同 sight 的比例，之後 log ---
                # 這裡會先拿到一個 list of sights，統計各個 sight 的比例
                sight_history = np.array(original_env.sight_history)
                unique, counts = np.unique(sight_history, return_counts=True)
                visibility_scalar = dict(zip(unique, counts))
                visibility_scalar = {k: v / len(sight_history) for k, v in visibility_scalar.items()}
                for k, v in visibility_scalar.items():
                    logger.log_stat(f"selected_sight_ratio/{k}s", v, runner.t_env)

            logger.console_logger.info(
                "t_env: {} / {}".format(runner.t_env, args.t_max)
            )
            logger.console_logger.info(
                "Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.t_env, args.t_max),
                    time_str(time.time() - start_time),
                )
            )
            last_time = time.time()

            last_test_T = runner.t_env
            # for _ in range(n_test_runs):
            runner.run(test_mode=True)

            if need_record_sight_history:
                # Reset sight_history after test
                original_env.sight_history = []

        if args.save_model and (
                runner.t_env - model_save_time >= args.save_model_interval
                or model_save_time == 0
        ):
            model_save_time = runner.t_env
            # save_path = os.path.join(
            #     # args.local_results_path, "models", args.unique_token, str(runner.t_env)
            #     dirname(dirname(abspath(__file__))), args.save_path, "models", args.unique_token, str(runner.t_env)
            # )
            model_direc = str(os.path.join(
                dirname(dirname(abspath(__file__))), args.save_path, "models"
            ))
            save_path = os.path.join(model_direc, args.unique_token, str(runner.t_env))
            # "results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        # Check policy difference periodically (by sampling from buffer)
        if (runner.t_env - last_policy_diff1_T) >= args.policy_diff_interval_1 > 0:
            if buffer.can_sample(args.batch_size):
                learner.check_policy_diff(buffer, runner.t_env, episode, interval_id=1)
                last_policy_diff1_T = runner.t_env
        if (runner.t_env - last_policy_diff2_T) >= args.policy_diff_interval_2 > 0:
            if buffer.can_sample(args.batch_size):
                learner.check_policy_diff(buffer, runner.t_env, episode, interval_id=2)
                last_policy_diff2_T = runner.t_env
        if (runner.t_env - last_policy_diff3_T) >= args.policy_diff_interval_3 > 0:
            if buffer.can_sample(args.batch_size):
                learner.check_policy_diff(buffer, runner.t_env, episode, interval_id=3)
                last_policy_diff3_T = runner.t_env

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

        # --- Dormant checking ---
        if (runner.t_env - last_dormant_T) >= args.dormant_check_interval > 0:
            if buffer.can_sample(args.batch_size):
                learner.check_dormant_neurons(buffer, runner.t_env)
                last_dormant_T = runner.t_env

        # Reset periodically
        next_reset_stop = runner.t_env + args.reset_freq >= args.t_max  # Prevent from testing the nn that is reset
        # TODO: this is not tested

        if preprocess_manager.reset_after_switch_visibility:
            whether_reset = preprocess_manager.is_time_to_reset_model(runner.t_env)
            # print(f'whether_reset: {whether_reset}')
        else:
            whether_reset = 0 < args.reset_freq <= runner.t_env - last_reset_T

        if whether_reset and not next_reset_stop:
            # assert buffer.can_sample(args.batch_size)
            if buffer.can_sample(args.batch_size):
                print(f"Resetting parameters; t_env: {runner.t_env}")
                learner.reset_param(args.reset_pos)
                last_reset_T = runner.t_env

                # Offline train
                for _ in range(args.n_offline_grad_steps):
                    episode_sample = preprocess_manager.sample_current_needed_data(buffer, args, runner.t_env)

                    # Truncate batch to only filled timesteps
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]

                    if episode_sample.device != args.device:
                        episode_sample.to(args.device)

                    learner.offline_train(episode_sample, runner.t_env, episode)

                # Distillation train
                for _ in range(args.n_distillation_grad_steps):
                    episode_sample = preprocess_manager.sample_current_needed_data(buffer, args, runner.t_env)

                    # Truncate batch to only filled timesteps
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]

                    if episode_sample.device != args.device:
                        episode_sample.to(args.device)

                    learner.distillation_train(episode_sample, runner.t_env, episode)

        # # Check dormant neurons and re-init periodically
        # if (0 < args.redo_freq <= runner.t_env - last_redo_T) and buffer.can_sample(args.batch_size):
        #     episode_sample = sample_batch_and_preprocess(buffer, preprocess_schedule_function(runner.t_env), args)
        #
        #     # Truncate batch to only filled timesteps
        #     max_ep_t = episode_sample.max_t_filled()
        #     episode_sample = episode_sample[:, :max_ep_t]
        #
        #     if episode_sample.device != args.device:
        #         episode_sample.to(args.device)
        #
        #     learner.estimate_dormant_and_reinit(episode_sample, runner.t_env, episode)

        if runner.t_env - learner.log_stats_t >= learner.args.learner_log_interval:
            learner.log_stats_t = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
                                          config["test_nepisode"] // config["batch_size_run"]
                                  ) * config["batch_size_run"]

    return config
