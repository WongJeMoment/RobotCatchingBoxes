# 使用 Isaac Sim 训练 Unitree G1

这是一个基于 **Isaac Lab 2.3.x + Isaac Sim 5.1 + RSL-RL PPO** 的 G1 人形机器人训练项目。行走和抗冲击任务使用 Isaac Lab 官方维护的 `G1_MINIMAL_CFG`；双手接箱任务改用带完整手部碰撞体的 `G1_CFG`。所有任务都在本仓库注册，不需要修改 Isaac Lab 源码。

项目提供基础 locomotion 任务和带箱体冲击的抗扰动任务：

| 任务 ID | 用途 |
| --- | --- |
| `Unitree-G1-Velocity-Flat-v0` | 平地训练，建议先跑 |
| `Unitree-G1-Velocity-Flat-Play-v0` | 平地评估 |
| `Unitree-G1-Velocity-Rough-v0` | 崎岖地形训练 |
| `Unitree-G1-Velocity-Rough-Play-v0` | 崎岖地形评估 |
| `Unitree-G1-Velocity-Flat-BoxThrow-v0` | 平地箱体冲击训练 |
| `Unitree-G1-Velocity-Flat-BoxThrow-Play-v0` | 平地箱体冲击评估 |
| `Unitree-G1-Velocity-Rough-BoxThrow-v0` | 崎岖地形箱体冲击训练 |
| `Unitree-G1-Velocity-Rough-BoxThrow-Play-v0` | 崎岖地形箱体冲击评估 |
| `Unitree-G1-Catch-Box-v0` | 固定基座双手接箱训练 |
| `Unitree-G1-Catch-Box-Play-v0` | 双手接箱可视化评估 |
| `Unitree-G1-WholeBody-Catch-Box-v0` | 全身站立平衡与双手接箱联合训练 |
| `Unitree-G1-WholeBody-Catch-Box-Play-v0` | 全身接箱 Demo 可视化 |

## 1. 环境要求

- Ubuntu 22.04/24.04 或 Windows 11
- NVIDIA RTX GPU；推荐至少 16 GB 显存、32 GB 内存
- Isaac Sim 5.1.0（Python 3.11）
- Isaac Lab v2.3.2

先按 [Isaac Lab 官方安装文档](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/index.html) 安装并验证 Isaac Sim 和 Isaac Lab。以下假设 Isaac Lab 位于 `~/IsaacLab`：

```bash
export ISAACLAB_PATH="$HOME/IsaacLab"
cd /path/to/RSS2027

# 可选；脚本本身也会把 source/ 加入 Python 路径
"$ISAACLAB_PATH/isaaclab.sh" -p -m pip install -e .
```

## 2. 快速冒烟测试

先只开 32 个并行环境、训练 2 次迭代，确认资产下载、仿真和 PPO 全部正常：

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/train.py \
  --num_envs 32 \
  --max_iterations 2 \
  --headless
```

首次启动会从 NVIDIA Nucleus 下载 G1 USD 和地形资产，所需时间取决于网络。

## 3. 正式训练

平地训练：

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/train.py \
  --task Unitree-G1-Velocity-Flat-v0 \
  --num_envs 4096 \
  --max_iterations 1500 \
  --seed 42 \
  --headless
```

崎岖地形训练：

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/train.py \
  --task Unitree-G1-Velocity-Rough-v0 \
  --num_envs 4096 \
  --max_iterations 3000 \
  --seed 42 \
  --headless
```

显存不足时把 `--num_envs` 降到 2048、1024 或 512。日志和 checkpoint 分别写入：

```text
logs/rsl_rl/unitree_g1_flat/<日期>_ppo/
logs/rsl_rl/unitree_g1_rough/<日期>_ppo/
```

通过 TensorBoard 查看曲线：

```bash
tensorboard --logdir logs/rsl_rl
```

## 4. 训练可视化

### Isaac Sim 实时画面

去掉 `--headless` 即可打开窗口。渲染会明显降低训练速度，建议实时观察时只启用少量并行环境：

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/train.py \
  --task Unitree-G1-Velocity-Flat-v0 \
  --num_envs 32 \
  --max_iterations 1500
```

默认相机会跟随第 0 个环境中的 G1。窗口主要用于检查机器人姿态、接触和地形；正式训练仍建议使用 `--headless`。

### TensorBoard 学习曲线

训练运行期间另开一个终端：

```bash
conda activate env_isaaclab
cd /path/to/RSS2027
tensorboard --logdir logs/rsl_rl --port 6006
```

浏览器访问 `http://localhost:6006`，重点关注 episode reward、episode length、速度跟踪奖励和 value/entropy 等指标。

### 训练期间定期录像

以下命令保持无窗口训练，每 2000 个仿真步录制一段 300 帧视频：

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/train.py \
  --task Unitree-G1-Velocity-Flat-v0 \
  --num_envs 1024 \
  --max_iterations 1500 \
  --headless \
  --video \
  --video_length 300 \
  --video_interval 2000
```

视频保存在对应运行目录的 `videos/train/` 下。录像需要 `ffmpeg`，同时会增加显存占用并降低训练速度。

## 5. 评估、录制与导出

评估最新一次平地训练；不要加 `--headless`，即可打开 Isaac Sim 窗口：

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/play.py \
  --task Unitree-G1-Velocity-Flat-Play-v0 \
  --num_envs 1
```

指定 checkpoint：

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/play.py \
  --task Unitree-G1-Velocity-Flat-Play-v0 \
  --checkpoint /absolute/path/to/model_1499.pt \
  --num_envs 1
```

录制 600 帧视频（适合无显示器服务器）：

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/play.py \
  --task Unitree-G1-Velocity-Flat-Play-v0 \
  --checkpoint /absolute/path/to/model_1499.pt \
  --video --video_length 600 --headless
```

`play.py` 会沿用 Isaac Lab 官方流程，同时在 checkpoint 旁的 `exported/` 目录导出 `policy.pt` 和 `policy.onnx`。

## 6. 修改任务

- 速度指令、并行环境数和评估设置：`source/g1_locomotion/tasks/velocity/g1_env_cfg.py`
- PPO 网络、迭代数和保存间隔：`source/g1_locomotion/tasks/velocity/agents/rsl_rl_ppo_cfg.py`
- Gymnasium 任务注册：`source/g1_locomotion/tasks/__init__.py`

建议先让平地策略收敛，再训练崎岖地形。当前代码输出的是仿真策略；将策略部署到真实 G1 前，还需要加入更完整的动力学随机化、控制频率/观测对齐、安全限幅，并在吊架或仿真硬件在环环境中验证。不要把未验证策略直接下发到真机。

## 7. 不同尺寸箱体冲击训练

箱体扰动任务会为每个环境创建三种可碰撞刚体：

| 尺寸 | 长宽高 | 质量 |
| --- | --- | --- |
| 小 | 0.15 × 0.15 × 0.15 m | 0.5 kg |
| 中 | 0.30 × 0.25 × 0.25 m | 1.5 kg |
| 大 | 0.45 × 0.35 × 0.35 m | 3.0 kg |

每隔 4–7 秒，每个并行环境会独立随机选择一种尺寸，并随机化投掷方位、距离、目标点、飞行时间、姿态和旋转速度。投掷速度使用弹道方程计算，使箱体大致命中 G1 的骨盆/躯干区域。

先用较少环境验证并观察投掷效果：

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/train.py \
  --task Unitree-G1-Velocity-Flat-BoxThrow-v0 \
  --num_envs 4 \
  --max_iterations 2 \
  'env.events.throw_box.interval_range_s=[0.1,0.1]'
```

末尾的 Hydra 参数只为快速观察而把投掷间隔临时改成 0.1 秒，不会修改正式配置文件。

正式训练：

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/train.py \
  --task Unitree-G1-Velocity-Flat-BoxThrow-v0 \
  --num_envs 1024 \
  --max_iterations 2000 \
  --seed 42 \
  --headless
```

训练完成后可视化评估：

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/play.py \
  --task Unitree-G1-Velocity-Flat-BoxThrow-Play-v0 \
  --num_envs 1
```

箱体数量会增加显存占用，建议从 1024 个环境开始。箱体参数位于 `g1_env_cfg.py`，投掷弹道位于 `mdp/box_throw.py`。箱体任务把终止条件从“躯干发生接触”改为“机器人倾倒超过约 57°”，从而允许策略学习受击后的恢复动作。

## 8. 双手抓取/接住抛来的箱体

`Unitree-G1-Catch-Box-v0` 是一个单独的第一阶段技能任务。每次 episode 会从机器人正前方随机抛出小、中、大三种箱体之一；策略控制躯干、双臂和手指共 25 个关节。策略观测包括箱体相对位姿与速度、尺寸 one-hot、双手相对运动、手部接触力和关节状态。

奖励分为连续的靠近/对中/速度匹配奖励，以及双手同时接触、保持高度和稳定抓取奖励。只有当双手接触、箱体位于两手之间、相对速度较低且离地，并连续保持 0.3 秒时，才记录 `box_caught` 成功；擦碰一下不会被当成抓取成功。

先做小规模冒烟测试：

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/train.py \
  --task Unitree-G1-Catch-Box-v0 \
  --num_envs 4 \
  --max_iterations 2 \
  --headless
```

正式训练（完整手部碰撞计算较重，16 GB 显存建议从 512 个环境开始）：

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/train.py \
  --task Unitree-G1-Catch-Box-v0 \
  --num_envs 512 \
  --max_iterations 3000 \
  --seed 42 \
  --headless
```

训练时查看 TensorBoard：

```bash
tensorboard --logdir logs/rsl_rl/unitree_g1_catch_box --port 6006
```

重点观察 `Episode_Termination/box_caught`、`Episode_Termination/box_dropped`、`Episode_Reward/catch_success` 和 `Episode_Reward/bilateral_contact`。成功率长期为零时，可先缩小 `CatchBoxEventsCfg` 中尺寸、投掷方位和速度的随机范围，再逐步放宽。

训练完成后打开 Isaac Sim 窗口评估最新 checkpoint：

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/play.py \
  --task Unitree-G1-Catch-Box-Play-v0 \
  --num_envs 1
```

接箱任务默认使用静态 `env` 相机：启动时会对准机器人，但不会在每个渲染帧强制回到默认视角，因此训练过程中可以自由旋转、平移和缩放 Viewport。若需要重新启用机器人跟随相机，可在 Isaac Lab 的 `Viewer Settings` 中把原点切换为 `Robot`。

也可以指定 checkpoint 并录制视频：

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/play.py \
  --task Unitree-G1-Catch-Box-Play-v0 \
  --checkpoint /absolute/path/to/model_2999.pt \
  --video --video_length 600 --headless
```

这个阶段有意固定 G1 基座，让 PPO 先利用仿真状态学会可靠的接箱协调；它还不是“边走边接”的全身策略，也没有接入相机图像。后续可把该策略作为上肢技能，与已经训练好的行走策略做分层组合或蒸馏，再开放根部和腿部自由度；若要用真实视觉，还需把箱体状态观测替换为相机与目标检测/位姿估计结果。

## 9. 上下半身联合的全身接箱 Demo

`Unitree-G1-WholeBody-Catch-Box-v0` 释放 G1 根节点并联合控制全部 37 个关节：下半身分支控制 12 个髋、膝、踝关节，上半身分支控制躯干、双臂和手指的 25 个关节。策略同时观察基座速度、重力方向、全身关节状态、箱体运动和双手接触力。奖励要求机器人保持站立、减少足底滑移并用双手稳定接箱；只有接住箱体且基座高度和倾角仍满足要求时才算成功。

先执行一轮冒烟测试：

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/train.py \
  --task Unitree-G1-WholeBody-Catch-Box-v0 \
  --num_envs 4 \
  --max_iterations 2 \
  --headless
```

正式训练建议从 256 个环境开始：

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/train.py \
  --task Unitree-G1-WholeBody-Catch-Box-v0 \
  --num_envs 256 \
  --max_iterations 5000 \
  --seed 42 \
  --headless
```

可视化最新策略：

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/play.py \
  --task Unitree-G1-WholeBody-Catch-Box-Play-v0 \
  --num_envs 1
```

日志位于 `logs/rsl_rl/unitree_g1_whole_body_catch_box/`。重点观察 `Episode_Termination/box_caught`、`Episode_Termination/bad_orientation`、`Episode_Termination/root_too_low`、`Episode_Reward/flat_orientation_l2` 和 `Episode_Reward/catch_success`。这是单策略、端到端的站立接箱 Demo；它允许冲击后的恢复步，但速度命令固定为零，因此尚不是边走边接。

## 10. Isaac Sim 到 MuJoCo 的 sim2sim

仓库内置了全身接箱策略的 MuJoCo runner。它不启动 Isaac Sim，而是在 MuJoCo 中复现训练时的 150 维观测、37 维动作、关节顺序、PD 增益、三种随机投掷箱体和 episode 终止条件。物理步长为 0.005 秒，策略每 4 个物理步执行一次，即 50 Hz。

建议单独使用轻量的 MuJoCo 环境，避免改动已经配置好的 Isaac Lab 环境：

```bash
conda create -n g1_sim2sim python=3.11 -y
conda activate g1_sim2sim
python -m pip install -e ".[sim2sim]"
```

使用本项目中的 `model_2999.pt` 打开 MuJoCo viewer：

```bash
python scripts/sim2sim_mujoco.py \
  --checkpoint "$(pwd)/logs/rsl_rl/unitree_g1_whole_body_catch_box/2026-08-30_21-01-43_ppo/model_2999.pt"
```

runner 可以直接加载 RSL-RL 的 `model_*.pt`，也可以加载 `play.py` 导出的 `exported/policy.pt`。不传 `--checkpoint` 时，会自动选择 `logs/rsl_rl/unitree_g1_whole_body_catch_box/` 下时间最新一次运行的最大迭代 checkpoint。viewer 中按 `R` 可重新投掷并复位。

为避免未限幅的 PPO 输出把脚踝直接推到机械限位，runner 默认将策略动作限制在 `[-1, 1]`，并使用与 Isaac G1 USD 相同尺寸和局部位姿的鞋底碰撞盒。若要严格复现本次训练中 `clip_actions: null` 的原始动作语义，可添加 `--action-clip 0`；该 checkpoint 的首帧腿部动作幅值可超过 5，关闭限幅时 MuJoCo 中的脚踝可能迅速失稳。

无窗口服务器先执行契约检查和短时冒烟测试：

```bash
python scripts/sim2sim_mujoco.py \
  --checkpoint /absolute/path/to/model_2999.pt \
  --dry-run

python scripts/sim2sim_mujoco.py \
  --checkpoint /absolute/path/to/model_2999.pt \
  --headless --duration 8
```

只检查 MuJoCo 模型和 PD 站立、不运行神经网络时使用：

```bash
python scripts/sim2sim_mujoco.py --zero-action --headless --duration 8
```

运行自动测试：

```bash
PYTHONPATH=source python -m unittest -v tests/test_sim2sim_contract.py
```

MuJoCo G1 模型来自 MuJoCo Menagerie 的 37 关节历史版本；该版本与 Isaac Sim 5.1 的 `G1_CFG` 命名和运动链匹配。资产来源与本地改动记录在 `assets/mujoco/unitree_g1/SOURCE.md`，BSD-3-Clause 许可保留在同目录的 `LICENSE`。

这里的策略仍使用箱体真值状态和仿真接触力，因此这是 physics sim2sim 验证，不是可直接上真机的视觉部署。MuJoCo 和 PhysX 的接触、摩擦和关节模型不同；策略在 Isaac Sim 成功但在 MuJoCo 失败，通常意味着训练随机化或策略鲁棒性还不足，应把 sim2sim 结果作为部署门槛，而不是绕过它直接下发真机。

## 常见问题

**提示找不到 Isaac Lab**

设置 `ISAACLAB_PATH`，或给脚本添加 `--isaaclab-path /path/to/IsaacLab`。后一个参数会在启动官方 runner 前被本项目移除。

**CUDA 显存不足**

减少 `--num_envs`；录制视频还会额外占用显存。

**G1 一开始会摔倒**

训练初期这是正常现象。冒烟测试只验证程序能运行，不会得到会行走的策略。平地任务的默认训练量是 1500 次 PPO 迭代。
