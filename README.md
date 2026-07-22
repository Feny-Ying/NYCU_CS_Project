# NYCU_CS_Project
##訓練指令
CUDA_VISIBLE_DEVICES=0 python3 main.py --config="cityflow/mappo/base" --env-config=cityflow with env_args.time_limit=50 seed=1 t_max=20000000 save_path="result_0305_cityflow_dsr" name="mappo with dsr" batch_size_run=2 batch_size=2 buffer_size=2 env_args.episode_limit=750 preprocess_desc="cityflow:3s->ucb(0s,1s,2s),c=2,w=5000"
在/config/envs/cityflow.yaml可以更改實驗的地圖

