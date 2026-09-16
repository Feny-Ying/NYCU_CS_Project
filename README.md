# ITM via Dynamic Sight Range in CityFlow Simulation

以 **Dynamic Sight Range (DSR)** 改善多路口號誌控制 (Intelligent Traffic Management, ITM) 的研究專題。
本專案在 [CityFlow](https://github.com/cityflow-project/CityFlow) 模擬器上，透過一個 bandit-based 的 meta-controller 動態決定每個 episode 中 agent 的觀測範圍 (sight range)，再交由 MARL (MAPPO) 進行號誌相位決策。

---

## 動機 (Motivation)

- **Sight range dilemma**：觀測範圍太小，agent 看不到即將到來的車流；範圍太大，觀測維度暴增且混入大量無關資訊，反而拖慢收斂、降低表現。最佳範圍會隨路網與車流而異，無法事先人工指定。
- **交通管理是 non-stationary 環境**：車流隨時間變化，固定的觀測設定難以長期維持最佳。
- 採用 **Centralized Training with Decentralized Execution (CTDE)** 架構，讓各路口 agent 在訓練時共享資訊、執行時僅依局部觀測行動。

本專案的核心問題：**能否讓系統自己學出「該看多遠」？**

---

## 方法 (Method)

### 整體架構

```
            Selected sight range d*
  ┌──────────────────┐ ───────────────▶ ┌──────────────────┐
  │  Meta-Controller │                  │   MARL (MAPPO)   │ ──▶ CityFlow Env
  │   (SW-UCB)       │ ◀─────────────── │  Q1 ... Qi ... QN │
  └──────────────────┘  episode return  └──────────────────┘
```

1. **Meta-Controller** 將可選的 sight range 視為 multi-armed bandit 的 arms，依 UCB 分數選出本 episode 使用的 `d*`。
2. **MARL** 在該 sight range 下與環境互動一整個 episode。
3. episode 結束後，以 episode return 回頭更新 bandit 的統計量。

### Bandit 演算法

- **UCB**：`UCB_t(a) = X̂_t(a) + c · sqrt(log t / N_t(a))`
- **SW-UCB (Sliding-Window UCB)**：只採計最近 `w` 個 episode 的 reward 與選擇次數，用以因應 non-stationary 的訓練過程。

### Observation 設計

Observation 依「**離路口的距離**」切片，每個 lane 在每個切片中給一個 **queue value**（該區段內等待車輛數）：

```
observation = [
    [agent_1, lane_1, lane_2, ...],
    [agent_2, lane_1, lane_2, ...],
    ...
]
```

Work flow：`global observation` → `slicing observation (by d*)` → `MARL algorithm`

> 早期版本使用 waiting value；後續改為 queue value，使 reward 與觀測定義一致。

### RL 定義

| 項目 | 定義 |
| --- | --- |
| **Action** | 每隔 interval `t` 決定下一個 step 使用的 traffic light phase（杭州路網預設 9 個 phase） |
| **State** | 當前 phase + lane-based 的等待 / 排隊車輛資訊（依 sight range 切片） |
| **Reward** | `reward = current total queue length − last total queue length`； |

Phase 說明（杭州路網）：

| Phase | 內容 |
| --- | --- |
| 1 | 所有方向右轉 (5s) |
| 2 | 水平方向直走 + 右轉，垂直方向右轉 |
| 3 | 垂直方向直走 + 右轉，水平方向右轉 |
| 4 | 水平方向左轉 + 右轉，垂直方向右轉 |
| 5 | 垂直方向左轉 + 右轉，水平方向右轉 |
| 6 / 7 | 所有方向右轉，水平朝右 / 朝左方向全開 |
| 8 / 9 | 所有方向右轉，垂直向上 / 向下方向全開 |

---

## CityFlow 設定檔

| 檔案 | 內容 |
| --- | --- |
| `roadnet.json` | **road**：points（道路端點）、lanes（道路寬度、速限）<br>**intersection**：roads（連接的道路）、road links（lane 連結資訊）、traffic light（light phases：`time` 固定時相持續時間、`availableRoadLinks` 綠燈的 road links） |
| `flow.json` | vehicle（車輛長寬、行駛速度）、route（行經路線）、interval（生成間隔）、startTime、endTime |
| `config.json` | interval（每個 step 的秒數）、roadnetFile、flowFile、`rlTrafficLight`、`saveReplay` 等 |

---

## 實驗設定 (Experiments)

**演算法**：MAPPO　**指標**：Average Travel Time (ATT)、Throughput、Simulator Time

### 測試路網與車流

| 城市 | 路口數 | 車流檔 | 流量 |
| --- | --- | --- | --- |
| 杭州 | 16 (4×4) | `anon_4_4_hangzhou_real` | 2,983 veh/hr |
| 杭州 | 16 (4×4) | `anon_4_4_hangzhou_real_5734` | 6,538 veh/hr |
| 杭州 | 16 (4×4) | `anon_4_4_hangzhou_real_5816` | 6,984 veh/hr |
| 濟南 | 12 (3×4) | `anon_3_4_jinan_real_2000` | 4,365 veh/hr |
| 濟南 | 12 (3×4) | `anon_3_4_jinan_real_2500` | 5,494 veh/hr |
| 濟南 | 12 (3×4) | `anon_3_4_jinan_real` | 6,295 veh/hr |
| 紐約 | 48 (16×3) | `anon_16_3_newyork_real` | 2,824 veh/hr |
| 紐約 | 196 (28×7) | `anon_28_7_newyork_real_double` | 11,058 veh/hr |
| 紐約 | 196 (28×7) | `anon_28_7_newyork_real_triple` | 16,337 veh/hr |


---

## 結果 (Results)

### Sight range 集合 `{10, 30, 50, 100} m`

| 路網 | Sight range 表現排序 | 最常被選中的範圍 |
| --- | --- | --- |
| 杭州 (real) | DSR > 50m > 100m > 30m > 10m | 50m |
| 杭州 (real_5816) | 50m > DSR > 100m > 30m > 10m | 50m |
| 濟南 (real) | 50m > DSR > 100m > 30m > 10m | 50m |
| 濟南 (real_2000) | 50m > DSR > 100m > 30m > 10m | 50m |

### Sight range 集合 `{50, 75, 100} m`

| 路網 | Sight range 表現排序 | 最常被選中的範圍 |
| --- | --- | --- |
| 杭州 (real) | DSR > 50m > 75m > 100m | 50m |
| 濟南 (real) | DSR > 50m > 75m > 100m | 50m |

### 結論

- **Robustness of DSR**：DSR 在各路網中皆能維持第一梯隊的表現，展現高度適應性。
- **Optimal Range Convergence**：訓練後期，DSR 會穩定收斂到該路網專屬的最佳 sight range。
- **Accurate Boundary Identification**：即使在未知環境中，也能找出合適的感知邊界。
- **Strategic Foundation**：為後續交通管理策略的設計與最佳化提供 data-driven 的基礎。

---

## 延伸實驗 (Extensions)

### 1. Flexible Sight Range

原始設計中，一個 episode 內所有 agent 共用同一個 sight range；新設計允許**各 agent 各自選擇不同的 sight range**。

- 路網：濟南 12 路口，`flow_real` 6,295 veh/hr
- 候選範圍：`{25, 50, 75, 100} m`
- 結果：new DSR 與 old DSR 表現接近，兩者在此設定下皆未超越固定 50m baseline，顯示放寬到 per-agent 選擇會擴大搜尋空間、增加學習難度。

---

## 專案結構

```
.
├── config/
│   └── algs/              # 演算法 設定檔 (mappo、qmix)
│       └── cityflow/
│   └── envs/              # CityFlow 設定檔 (roadnet / flow / config)
│       ├── Hangzhou/
│       ├── Jinan/
│       ├── NewYork/
│       └── SH1/
├── epy_tools/       # UCB / SW-UCB meta-controller
├── runner/               # 訓練與評估腳本
└── learners/              # 演算法程式碼
```

---

## 使用方式

### 環境需求

```bash
# 安裝 CityFlow
git clone https://github.com/cityflow-project/CityFlow.git
cd CityFlow && pip install .

# 安裝本專案相依套件
pip install -r requirements.txt
```

### 訓練

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
```

### 固定 sight range 的 baseline

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
  preprocess_desc="cityflow:3s->ucb(0s),c=2,w=5000"
```

---

## 主要指標說明

| 指標 | 說明 |
| --- | --- |
| **ATT (Average Travel Time)** | 車輛平均旅行時間，越低越好 |
| **Throughput** | 單位時間內完成行程的車輛數，越高越好 |

---

## 參考

- CityFlow: A Multi-Agent Reinforcement Learning Environment for Large Scale City Traffic Scenario
- MAPPO: The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games
- Sliding-Window UCB for non-stationary bandit problems

## 作者

蔡浚庭、劉逢穎
