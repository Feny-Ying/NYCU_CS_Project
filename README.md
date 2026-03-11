# NYCU_CS_Project
command CUDA_VISIBLE_DEVICES=0 python3 main.py --config="cityflow/qmix/base" --env-config=cityflow with env_args.time_limit=50 seed=1 t_max=5300000 save_path="result_1016_cityflow" name="qmix 1s"

CUDA_VISIBLE_DEVICES=0 python3 main.py --config="cityflow/qmix/base" --env-config=cityflow with env_args.time_limit=50 seed=1 t_max=5300000 save_path="result_1016_cityflow" name="qmix 1s" batch_size=8 hidden_dim=128 env_args.episode_limit=1000 buffer_size=2000

CUDA_VISIBLE_DEVICES=0 python3 main.py --config="cityflow/mappo/base" --env-config=cityflow with env_args.time_limit=50 seed=1 t_max=10000 save_path="result_1115_mappo_evaluate_cityflow" name="mappo 1s" batch_size=8 hidden_dim=128 env_args.episode_limit=3000 buffer_size=2000 test_nepisode=2

CUDA_VISIBLE_DEVICES=0 python3 main.py --config="cityflow/qmix/base" --env-config=cityflow with env_args.time_limit=50 seed=1 t_max=1000 save_path="result_1122_cityflow_evaluate_cityflow " name="qmix 1s" batch_size=8 hidden_dim=128 env_args.episode_limit=200 buffer_size=2000 runner_log_interval=200 evaluate=true test_nepisode=5 checkpoint_path="/home/bluedyee/Kimo/result_1121_qmix_cityflow/models/qmix 1s_seed1_config/envs/Hangzhou/4_4/config.json_2025-11-20 20:57:23.749831/"

CUDA_VISIBLE_DEVICES=0 python3 main.py --config="cityflow/qmix/base" --env-config=cityflow with env_args.time_limit=50 seed=1 t_max=20000000 save_path="result_1229_cityflow_cityflow_1s_2s_3s" name="qmix with dsr 1s 2s 3s" batch_size=8 hidden_dim=128 env_args.episode_limit=200 buffer_size=2000 preprocess_desc="cityflow:3s->ucb(1s,2s,3s),c=2,w=5000"

CUDA_VISIBLE_DEVICES=0 python3 main.py --config="cityflow/qmix/base" --env-config=cityflow with env_args.time_limit=50 seed=1 t_max=20000000 save_path="result_0115d__cityflow_1s_2s_3s" name="qmix with dsr 1s 2s 3s" batch_size=8 hidden_dim=128 env_args.episode_limit=250 epsilon_anneal_time=400000 target_update_interval_or_tau=100 lr=0.00005 buffer_size=2000 gamma=0.9 standardise_rewards=False preprocess_desc="cityflow:3s->ucb(1s,2s,3s),c=2,w=5000"

CUDA_VISIBLE_DEVICES=0 python3 main.py --config="cityflow/mappo/base" --env-config=cityflow with env_args.time_limit=50 seed=1 t_max=20000000 save_path="result_0305_cityflow_dsr" name="mappo with dsr" batch_size_run=2 batch_size=2 buffer_size=2 env_args.episode_limit=750 preprocess_desc="cityflow:3s->ucb(0s,1s,2s),c=2,w=5000"
