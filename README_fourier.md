# SONIC VLA + Fourier 手 (sonicvla_fourier)

基于 GR00T 的视觉-语言-动作模型，面向宇树 G1 人形机器人 + Fourier FDH-6 灵巧手。

本项目在 [NVIDIA GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) 基础上集成了 Fourier FDH-6 灵巧手，实现 76 维动作控制（64D 运动令牌 + 12D Fourier 手部关节，每只手 6D）。

## 系统架构

- **动作空间**：76D = 64D 运动令牌（身体） + 6D 左手 + 6D 右手
- **灵巧手**：Fourier FDH-6（每只手 6 自由度：thumb_yaw, thumb_pitch, index, middle, ring, pinky）
- **机器人**：宇树 G1（含腰部，上躯干 + 下躯干）
- **策略模型**：基于 GR00T N1 微调

## 快速开始

### 环境要求

- Ubuntu 22.04 / 24.04
- Python 3.10
- CUDA GPU（用于策略服务器）
- [uv](https://docs.astral.sh/uv/) 包管理器
- tmux (`sudo apt install tmux`)
- [Git LFS](https://git-lfs.com/) (`sudo apt install git-lfs`)

### 1. 安装

```bash
# 克隆仓库
git lfs install                          # 确保已启用 LFS
git clone https://github.com/DavidSunok/sonicvla_fourier.git
cd sonicvla_fourier

# 下载上游模型权重（planner、motion data 等，约 2-3 GB）
# 这些大文件在上游仓库维护，需要从上游拉取
git remote add upstream https://github.com/NVlabs/GR00T-WholeBodyControl.git
git lfs fetch upstream main
git lfs checkout

# 安装推理环境
bash install_scripts/install_inference.sh
```

> 如果 `git lfs fetch` 较慢，可以设置代理：`export https_proxy=http://127.0.0.1:7897`

### 2. 启动策略服务器（GPU 服务器端）

在 GPU 服务器上，用你微调好的 checkpoint 启动 Isaac-GR00T PolicyServer：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python gr00t/eval/run_gr00t_server.py \
    --model-path ./outputs/fourier_finetune_v3/checkpoint-16000 \
    --embodiment-tag UNITREE_G1_SONIC_FOURIER \
    --device cuda:0 \
    --port 5550
```

> `embodiment-tag` 必须和微调时使用的标签一致。Fourier 手请使用 `UNITREE_G1_SONIC_FOURIER`。

### 3. 运行推理（机器人本地端）

在连接 G1 机器人和 Fourier 手的本地控制电脑上：

```bash
# 一键启动（在 tmux 中同时启动部署 + VLA 推理 + 键盘控制）
python gear_sonic/scripts/launch_inference.py \
    --policy-host <GPU服务器IP> \
    --policy-port 5550 \
    --hand-type fourier \
    --prompt "pick up the cup"
```

也可以分别手动启动各组件：

```bash
# 激活推理环境
source .venv_inference/bin/activate

# 终端 1：VLA 推理
python gear_sonic/scripts/run_vla_inference.py \
    --host <GPU服务器IP> \
    --port 5550 \
    --embodiment-tag unitree_g1_sonic_fourier \
    --hand-type fourier \
    --prompt "pick up the cup"

# 终端 2：C++ 部署（身体控制）
cd gear_sonic_deploy
./deploy.sh --disable-hands real
```

### 4. 键盘控制

| 按键 | 功能 |
|------|------|
| `k` | 启动/停止 C++ 控制循环 |
| `p` | 暂停/继续策略推理 |
| `i` | 移动到初始姿态 |
| `[` / `]` | 切换左手/右手张开/闭合 |
| `t <文本>` | 运行时切换语言指令 |

## 配置选项

### Fourier 手参数

```bash
python gear_sonic/scripts/run_vla_inference.py \
    --hand-type fourier \              # 使用 Fourier FDH-6 灵巧手
    --fourier-sim-mode \               # 仿真模式（不需要硬件）
    --fourier-action-scale 0.5         # 动作平滑系数（0.0-1.0，默认 1.0）
```

### 仿真模式（MuJoCo）

```bash
python gear_sonic/scripts/launch_inference.py \
    --sim \
    --hand-type fourier \
    --fourier-sim-mode
```

## 微调

用你自己的 Fourier 手数据微调 GR00T 模型：

1. 使用 GR00T 数据采集流程收集遥操作数据
2. 确保数据包含 Fourier 手部关节字段（`left_hand_fourier_joints`、`right_hand_fourier_joints`）
3. 配置 embodiment 标签为 `UNITREE_G1_SONIC_FOURIER`
4. 使用标准 GR00T 训练脚本进行微调

Fourier 手微调的动作空间为 74D 策略输出：`motion_token`(64) + `left_fourier_hand_joints`(5, thumb_yaw 被掩码) + `right_fourier_hand_joints`(5) = 74D，推理时填充 thumb_yaw 至 76D。

## 项目结构

```
sonicvla_fourier/
├── gear_sonic/
│   ├── data/
│   │   ├── features_sonic_vla.py        # 数据集特征与模态配置
│   │   └── robot_model/                 # 机器人模型（G1 + Fourier 手）
│   ├── scripts/
│   │   ├── launch_inference.py          # 一键 tmux 启动器
│   │   └── run_vla_inference.py         # VLA 推理循环
│   └── utils/inference/
│       └── fourier_inference_hand.py    # Fourier FDH-6 驱动
├── gear_sonic_deploy/                   # C++ 部署（身体控制）
├── install_scripts/
│   └── install_inference.sh             # 推理环境安装脚本
└── decoupled_wbc/                       # 全身控制器
```

## 致谢

- [NVIDIA GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) — 基础框架
- [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) — GR00T 模型
- [傅利叶智能](https://www.fourierintelligence.com/) — FDH-6 灵巧手硬件

## 许可证

本项目继承上游 GR00T-WholeBodyControl 仓库的许可证。
