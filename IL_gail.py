import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
import random

from tf2rl.experiments.trainer import Trainer
import os
import gym
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Dense, Input
from tf2rl.algos.ppo_1 import PPO
from tf2rl.algos.gail import GAIL
from tf2rl.algos.gail_1 import GAIL_1
from tf2rl.experiments.ildrl_trainer import ILDRL_Trainer
from tf2rl.experiments.gail_trainer import GAIL_Trainer


from smarts.env.hiway_env import HiWayEnv
from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface
from smarts.core.agent_interface import NeighborhoodVehicles, RGB
from smarts.core.controllers import ActionSpaceType


#### Load expert trajectories ####
def load_expert_trajectories(filepath):

    filenames  =  glob.glob(filepath)             # 返回所有匹配的文件路径列表

    trajectories  =  []
    for filename in filenames:
        trajectories.append(np.load(filename))      # 读文件内容

    obses  =  []
    next_obses  =  []
    actions  =  []

    for trajectory in trajectories:                 # 轨迹的长度为examples，这里取40

        obs  =  trajectory['obs']                   # (x, 80, 80, 9)
        action  =  trajectory['act']                # (x, 2)

        for i in range(obs.shape[0]-1):
            obses.append(obs[i])                                    # x个（80，80，9）
            next_obses.append(obs[i+1])
            act  =  action[i]
            act[0] +=  random.normalvariate(0, 0.1)                 # speed, 返回随机正态分布浮点数
            act[0]  =  np.clip(act[0], 0, 10)                       # 输入的数组，限定最小值及最大值分别为0和10
            act[0]  =  2.0 * ((act[0] - 0) / (10 - 0)) - 1.0        # 规范化为[-1, 1]
            act[1] +=  random.normalvariate(0, 0.1)                 # lane change
            act[1]  =  np.clip(act[1], -1, 1)
            actions.append(act)

    expert_trajs  =  {'obses': np.array(obses, dtype = np.float32),
                    'next_obses': np.array(next_obses, dtype = np.float32),
                    'actions': np.array(actions, dtype = np.float32)}

    return expert_trajs

# read args
parser  =  Trainer.get_argument()
parser.add_argument('file_path')                                    # expert_data/left_turn
parser.add_argument('--samples', type = int, default = 40)
args  =  parser.parse_args()

args.max_steps  =  10e4             # 100000
args.save_summary_interval  = 64
args.use_prioritized_rb  =  False
args.algo = 'Expert_IL_DRL'
args.horizon = 512
args.n_epoch = 10

args.scenario = args.file_path.split('/')[-1]

args.logdir  =  f'./train_results/{args.scenario}/{args.algo}'

# GPU setting
gpus  =  tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# create directory
if not os.path.exists('./gail_model_2/{}_{}'.format(args.file_path.split('/')[-1], args.samples)):
    os.makedirs('./gail_model_2/{}_{}'.format(args.file_path.split('/')[-1], args.samples))

# set up ensemble
state_shape  =  (80, 80, 9)
action_dim  =  2

# load and process data
OBS  =  []
ACT  =  []

files  =  glob.glob(args.file_path + '/*.npz')
if args.samples < len(files):
    files  =  random.sample(files, args.samples)        # 从files中随机获取samples个

for file in files:
    obs  =  np.load(file)['obs']                        # 获取观察的状态, (x, 80, 80, 9)
    act  =  np.load(file)['act']                        # 获取动作

    for i in range(obs.shape[0]):               # x
        OBS.append(obs[i])
        act[i, 0] +=  random.normalvariate(0, 0.1) # add a small noise to speed
        act[i, 0]  =  np.clip(act[i, 0], 0, 10)
        act[i, 0]  =  2.0 * ((act[i, 0] - 0) / (10 - 0)) - 1.0 # normalize speed, 规范化为[-1, 1]
        act[i, 1] +=  random.normalvariate(0, 0.1) # add a small noise to lane change, which does not affect the decision
        ACT.append(act[i])

OBS  =  np.array(OBS, dtype = np.float32)           # (x累加，80, 80,9)
ACT  =  np.array(ACT, dtype = np.float32)           # (x累加，, 2)

# model training
epochs  =  10
EPS  =  1e-6

#### Environment specs ####
ACTION_SPACE  =  gym.spaces.Box(low = -1.0, high = 1.0, shape = (2,))               # 连续对象，动作空间是个2维的向量，每个维度的取值区间都是[-1.0,1.0]
OBSERVATION_SPACE  =  gym.spaces.Box(low = 0, high = 1, shape = (80, 80, 9))        # 最小和最大值
AGENT_ID  =  'Agent-007'
states  =  np.zeros(shape = (80, 80, 9))                # 鸟瞰图像的大小为80×80×3,三个连续时间步鸟瞰图像

# observation space
def observation_adapter(env_obs):
    global states

    new_obs  =  env_obs.top_down_rgb[1] / 255.0                         # top_down_rgb[1] = (80, 80, 3), len(env_obs.top_down_rgb) = 2, top_down_rgb[0]为一些元数据，例如长宽、相机位置、朝向等

    states[:, :, 0:3]  =  states[:, :, 3:6]
    states[:, :, 3:6]  =  states[:, :, 6:9]
    states[:, :, 6:9]  =  new_obs

    if env_obs.events.collisions or env_obs.events.reached_goal:                # 如果发生碰撞或者到达目的地，重置状态为全零
        states  =  np.zeros(shape = (80, 80, 9))

    return np.array(states, dtype = np.float32)

# reward function
def reward_adapter(env_obs, env_reward):
    progress  =  env_obs.ego_vehicle_state.speed * 0.1

    goal  =  1 if env_obs.events.reached_goal else 0
    crash  =  -1 if env_obs.events.collisions else 0

    if args.algo  ==  'value_penalty' or args.algo  ==  'policy_constraint':
        return goal + crash
    else:
        return 0.01 * progress + goal + crash

# action space
def action_adapter(model_action):                       # 适配器
    speed  =  model_action[0]                           # output (-1, 1)
    speed  =  (speed - (-1)) * (10 - 0) / (1 - (-1))    # scale to (0, 10)
    speed  =  np.clip(speed, 0, 10)                     # 感觉有点多余
    # speed = 2.0 * ((speed - 0) / (10 - 0)) - 1.0        # 新加

    model_action[1]  =  np.clip(model_action[1], -1, 1) #

    # discretization
    if model_action[1] < -1/3:
        lane  =  -1                 # 向左车道改变动作
    elif model_action[1] > 1/3:
        lane  =  1                  # 向右车道改变动作
    else:
        lane  =  0                  # 车道保持动作

    return (speed, lane)            # 返回速度、车道

# information
def info_adapter(observation, reward, info):                # 允许对额外的信息进行进一步处理
    return info

# define scenario
if args.scenario  ==  'left_turn':
    scenario_path  =  ['scenarios/left_turn']
    max_episode_steps  =  400
elif args.scenario  ==  'roundabout':
    scenario_path  =  ['scenarios/roundabout']
    max_episode_steps  =  600
else:
    raise NotImplementedError

# define agent interface
agent_interface  =  AgentInterface(                             # 指定了智能体期望从环境中获得的观察结果以及智能体对环境所作的动作
    max_episode_steps = max_episode_steps,
    waypoints = True,
    neighborhood_vehicles = NeighborhoodVehicles(radius = 60),
    rgb = RGB(80, 80, 32/80),
    action = ActionSpaceType.LaneWithContinuousSpeed,
)

# define agent specs
agent_spec  =  AgentSpec(
    interface = agent_interface,
    observation_adapter = observation_adapter,
    reward_adapter = reward_adapter,
    action_adapter = action_adapter,
    info_adapter = info_adapter,
)

expert_trajs  =  load_expert_trajectories(args.file_path + '/*.npz')

for idx in range(1, 6):
    print(' =====  Training Ensemble Model {}  ===== '.format(idx))
    tf.random.set_seed(random.randint(1, 1000)) # reset tensorflow random seed
    np.random.seed(random.randint(1, 1000)) # reset numpy random seed

    # start training
    # create env
    env  =  HiWayEnv(scenarios = scenario_path, agent_specs = {AGENT_ID: agent_spec}, headless = True, seed = i)
    env.observation_space  =  OBSERVATION_SPACE
    env.action_space  =  ACTION_SPACE
    env.agent_id  =  AGENT_ID

    if args.algo == 'Expert_IL_DRL':
        policy = PPO(state_shape = OBSERVATION_SPACE.shape, action_dim = ACTION_SPACE.high.size, max_action = ACTION_SPACE.high[0],
                    batch_size = 32, clip_ratio = 0.2, n_epoch = 10, entropy_coef = 0.01, horizon = 512)
        irl = GAIL(state_shape = env.observation_space.shape, action_dim = env.action_space.high.size, batch_size = 32, n_training = 10)
        trainer = ILDRL_Trainer(idx, policy, env, args, irl, expert_trajs["obses"], expert_trajs["next_obses"], expert_trajs["actions"])

        # gail = GAIL_1(state_shape = env.observation_space.shape, action_dim = env.action_space.high.size, batch_size = 32, n_training = 10, horizon = 512)
        # trainer = GAIL_Trainer(idx, gail, env, args, expert_trajs["obses"], expert_trajs["next_obses"], expert_trajs["actions"])
    else:
        raise NotImplementedError

    # begin training
    trainer()
    # close env
    env.close()
