import torch
import numpy as np
from matplotlib import pyplot as plt
import argparse
from utils.replay_buffer import ReplayBuffer
from utils.matd3 import MATD3
import copy
from gym_pybullet_drones.new_envs.CircleSpread_Camera import CircleCameraAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

Env_name = 'circle'
action = 'vel'


class Runner:
    def __init__(self, args):
        self.args = args
        self.args.decive = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args.discrete = True
        self.env_name = Env_name
        self.number = args.N_drones
        self.seed = 1145  # 保证一个seed，名称使用记号--mark
        self.mark = args.mark
        self.load_mark = args.mark
        self.args.share_prob = 0.05  # 还是别共享了，有些无用
        self.args.obs_type = 'rgb'  # kin_target, rgb
        self.test_times = 3
        # Create env
        Ctrl_Freq = args.Ctrl_Freq  # 30
        self.env = CircleCameraAviary(gui=True, num_drones=args.N_drones, obs=ObservationType(self.args.obs_type),
                                      act=ActionType(action),
                                      need_target=True, obs_with_act=True, discrete=self.args.discrete)
        self.timestep = 1 / Ctrl_Freq  # 计算每个步骤的时间间隔 0.003

        if self.args.discrete:
            self.args.obs_rgb_dim_n, self.args.obs_other_dim_n = self.env.observation_space
        else:
            if ObservationType(self.args.obs_type) == ObservationType.RGB:  # obs_space:Box(0, 255, (3,48,64,4))
                self.args.obs_rgb_dim_n, self.args.obs_other_dim_n = self.env.observation_space
            elif ObservationType(self.args.obs_type) == ObservationType.KIN_target:
                self.args.obs_dim_n = [self.env.observation_space[i].shape[0] for i in range(self.env.NUM_DRONES)]
            else:
                raise ValueError("Unsupported observation type")
        if self.args.discrete:
            self.args.action_dim_n = self.env.action_space
        else:
            self.args.action_dim_n = [self.env.action_space[i].shape[0] for i in
                                      range(self.args.N_drones)]  # actions dimensions of N agents

        if ObservationType(self.args.obs_type) == ObservationType.RGB:
            print(f"obs_rgb_dim_n={self.args.obs_rgb_dim_n}, obs_other_dim_n={self.args.obs_other_dim_n}")
        elif ObservationType(self.args.obs_type) == ObservationType.KIN_target:
            print(f"obs_dim_n={self.args.obs_dim_n}")
        print(f"action_dim_n={self.args.action_dim_n}")

        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Create N agents
        print("Algorithm: MATD3")
        self.agent = MATD3(args)
        self.replay_buffer = ReplayBuffer(self.args)
        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0
        self.noise_std = self.args.noise_std_init  # Initialize noise_std

        # Load Model
        for agent_id in range(self.args.N_drones):
            # 加载模型参数    todo 修改
            model_path = "./model/{}/{}_actor_mark_{}_number_{}_step_{}k.pth".format(self.env_name,
                                                                                     self.args.algorithm,
                                                                                     self.load_mark,
                                                                                     self.number,
                                                                                     int(900))  # agent_id
            self.agent.actor.load_state_dict(torch.load(model_path))

    def run(self, ):
        for i in range(self.test_times):
            self.evaluate_policy()
        self.env.close()

    def evaluate_policy(self):
        all_states = []
        all_actions = []
        all_rewards = []

        for eval_time in range(self.args.evaluate_times):
            rgb_n, obs_n, _ = self.env.reset()
            pixels = self.agent.preprocess_rgb(rgb_n)
            episode_return = [0 for _ in range(self.args.N_drones)]
            episode_states = []
            episode_actions = []
            episode_rewards = []

            for _ in range(self.args.episode_limit):
                a_n = self.agent.choose_action(pixels, obs_n, noise_std=self.noise_std)  # (3,48,64,4) -> (3，4)
                rgb_next_n, obs_next_n, r_n, done_n, _, _ = self.env.step(copy.deepcopy(a_n))
                pixels_next = self.agent.preprocess_rgb(rgb_next_n)

                self.replay_buffer.store_transition(pixels, obs_n, a_n, r_n, pixels_next, obs_next_n, done_n)
                pixels, obs_n = pixels_next, obs_next_n
                self.total_steps += 1
                for i in range(self.args.N_drones):
                    episode_return[i] += r_n[i]

                # 保存状态、动作和奖励
                episode_states.append(obs_n)
                episode_actions.append(a_n)
                episode_rewards.append(r_n)

                obs_n = obs_next_n
                if all(done_n):
                    break

            all_states.append(episode_states)
            all_actions.append(episode_actions)
            all_rewards.append(episode_rewards)
            print("eval_time:{} \t episode_reward:{} \t".format(eval_time, episode_return))

        # 将数据转换为numpy数组
        all_states = np.array(all_states)
        all_actions = np.array(all_actions)
        all_rewards = np.array(all_rewards)

        # 绘制图
        for eval_time in range(self.args.evaluate_times):
            self.plot_results(all_states[eval_time], all_actions[eval_time], all_rewards[eval_time])

    def plot_results(self, states, actions, rewards):
        # 创建图
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 计算奖励范围以用于颜色映射
        min_reward = np.min(rewards)
        max_reward = np.max(rewards)

        # 定义目标点
        target = self.env.TARGET_POS

        # 绘制目标点
        ax.scatter(target[0], target[1], target[2], color='k', s=100, label='Target')

        # 生成三种颜色映射
        cmaps = [plt.cm.Reds, plt.cm.Greens, plt.cm.Blues]  # 三种渐变色
        norms = [plt.Normalize(min_reward - 1, max_reward) for _ in range(3)]

        for agent_id in range(self.args.N_drones):
            agent_states = states[:, agent_id, :3]
            agent_rewards = rewards[:, agent_id]
            cmap = cmaps[agent_id % len(cmaps)]  # 循环使用渐变色
            norm = norms[agent_id % len(norms)]
            step_size = max(1, len(agent_states) // 500)  # 绘制最多 500 个点

            # 绘制轨迹，根据奖励值调整颜色亮度
            for i in range(0, len(agent_states) - 1, step_size):
                color_intensity = norm(agent_rewards[i])
                line_color = cmap(color_intensity)
                ax.plot(agent_states[i:i + 2, 0], agent_states[i:i + 2, 1], agent_states[i:i + 2, 2],
                        color=line_color)

        ax.set_title('Agent Positions and Actions Over Time')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MADDPG and MATD3 in MPE environment")
    parser.add_argument("--max_train_steps", type=int, default=int(1e6), help=" Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=1000, help="Maximum number of steps per episode")
    parser.add_argument("--test_episode_limit", type=int, default=1000, help="Maximum number of steps per test episode")
    parser.add_argument("--evaluate_freq", type=float, default=150000,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=1, help="Evaluate times")
    parser.add_argument("--max_action", type=float, default=1.0, help="Max action")

    parser.add_argument("--algorithm", type=str, default="MATD3", help="MADDPG or MATD3")
    parser.add_argument("--buffer_size", type=int, default=int(1e5), help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")  # 1024-》4048
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--noise_std_init", type=float, default=0.025, help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_std_min", type=float, default=0.005, help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_decay_steps", type=float, default=3e5,
                        help="How many steps before the noise_std decays to the minimum")
    parser.add_argument("--use_noise_decay", type=bool, default=True, help="Whether to decay the noise_std")
    parser.add_argument("--lr_a", type=float, default=5e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=5e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="Softly update the target network")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Orthogonal initialization")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip")
    parser.add_argument("--device", type=str, default="cpu", help="Device to train model")
    # --------------------------------------MATD3--------------------------------------------------------------------
    parser.add_argument("--policy_noise", type=float, default=0.2, help="Target policy smoothing")
    parser.add_argument("--noise_clip", type=float, default=0.5, help="Clip noise")
    parser.add_argument("--policy_update_freq", type=int, default=2, help="The frequency of policy updates")

    parser.add_argument("--mark", type=int, default=1147, help="The frequency of policy updates")
    parser.add_argument("--N_drones", type=int, default=2, help="The number of drones")
    parser.add_argument("--Ctrl_Freq", type=int, default=30, help="The frequency of ctrl")
    args = parser.parse_args()
    args.noise_std_decay = (args.noise_std_init - args.noise_std_min) / args.noise_decay_steps

    # todo change mark !!!!!!
    runner = Runner(args)
    runner.run()
