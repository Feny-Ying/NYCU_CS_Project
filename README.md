# 🚦 NYCU CS Project: CityFlow MAPPO Experiment

本專案基於 **MAPPO (Multi-Agent PPO)** 演算法，針對 **CityFlow** 交通訊號控制模擬環境進行多 Agent 強化學習訓練與實驗。

---

## ⚙️ 環境設定與設定檔說明

實驗地圖與相關環境參數可於設定檔中進行修改：

* **地圖與環境設定檔路徑**：
  ```path
  /config/envs/cityflow.yaml

## ⚙️ 實驗指令

* **訓練指令**：
  ```bash
  CUDA_VISIBLE_DEVICES=0 python3 main.py \
  --config="cityflow/mappo/base" \
  --env-config=cityflow \
  with \
  env_args.time_limit=50 \
  seed=1 \
  t_max=20000000 \
  save_path="result_0305_cityflow_dsr" \
  name="mappo with dsr" \
  batch_size_run=2 \
  batch_size=2 \
  buffer_size=2 \
  env_args.episode_limit=750 \
  preprocess_desc="cityflow:3s->ucb(0s,1s,2s),c=2,w=5000"



