import torch
import numpy as np
import argparse
from utils.maddpg import MADDPG
from utils.matd3 import MATD3
import copy
from gym_pybullet_drones.envs.Spread3d import Spread3dAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
import matplotlib.pyplot as plt


Env_name = 'spread3d'
action = 'vel'


class Runner:
    def __init__(self, args):
        self.args = args
        self.args.decive = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env_name = Env_name
        self.number = 3  # todo 指定number
        self.seed = 1145  # 保证一个seed，名称使用记号--mark
        self.mark = 166  # todo 指定mark
        Load_Steps = 2900000  # self.args.max_train_steps = 1e6
        Ctrl_Freq = 30  # todo check
        # Create env
        self.env_evaluate = Spread3dAviary(gui=True, num_drones=args.N_drones, obs=ObservationType('kin_target'),
                                           act=ActionType(action),
                                           ctrl_freq=Ctrl_Freq,  # 这个值越大，仿真看起来越慢，应该是由于频率变高，速度调整的更小了
                                           need_target=True, obs_with_act=True)
        self.timestep = 1 / Ctrl_Freq  # 计算每个步骤的时间间隔 0.003

        self.args.obs_dim_n = [self.env_evaluate.observation_space[i].shape[0] for i in
                               range(self.args.N_drones)]  # obs dimensions of N agents
        self.args.action_dim_n = [self.env_evaluate.action_space[i].shape[0] for i in
                                  range(self.args.N_drones)]  # actions dimensions of N agents
        # print("observation_space=", self.env.observation_space)
        print("obs_dim_n={}".format(self.args.obs_dim_n))
        # print("action_space=", self.env.action_space)
        print("action_dim_n={}".format(self.args.action_dim_n))

        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Create N agents
        if self.args.algorithm == "MADDPG":
            print("Algorithm: MADDPG")
            self.agent_n = [MADDPG(args, agent_id) for agent_id in range(args.N_drones)]
        elif self.args.algorithm == "MATD3":
            print("Algorithm: MATD3")
            self.agent_n = MATD3.initialize_agents(args)
        else:
            print("Wrong!!!")

        for agent_id in range(self.args.N_drones):
            # 加载模型参数
            model_path = "./model/{}/{}_actor_mark_{}_number_{}_step_{}k_agent_{}.pth".format(self.env_name,
                                                                                              self.args.algorithm,
                                                                                              self.mark, self.number,
                                                                                              int(Load_Steps / 1000),
                                                                                              agent_id)
            self.agent_n[agent_id].actor.load_state_dict(torch.load(model_path))
        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.noise_std = self.args.noise_std_init  # Initialize noise_std

    def run(self, ):
        self.evaluate_policy()
        self.env_evaluate.close()

    def convert_obs_dict_to_array(self, obs_dict):
        obs_array = []
        if self.args.N_drones != 1:
            for i in range(self.args.N_drones):
                obs = obs_dict[i]
                # action_buffer_flat = np.hstack(obs['action_buffer'])    # 拉成一维
                obs_array.append(np.hstack([
                    obs['pos'],
                    obs['rpy'],
                    obs['vel'],
                    obs['ang_vel'],
                    obs['target_pos'],
                    obs['other_pos'],
                    obs['action_buffer']  # 先不考虑动作
                ]))
        else:
            pass
        return np.array(obs_array).astype('float32')

    def convert_wrap(self, obs_dict):
        if isinstance(obs_dict, dict):
            obs_dict = self.convert_obs_dict_to_array(obs_dict)
        else:
            obs_dict = obs_dict
        return obs_dict

    def evaluate_policy(self):
        evaluate_reward = 0
        all_states = []
        all_actions = []
        all_rewards = []

        for eval_time in range(self.args.evaluate_times):
            obs_n, _ = self.env_evaluate.reset()
            obs_n = self.convert_wrap(obs_n)
            episode_return = [0 for _ in range(self.args.N_drones)]
            episode_states = []
            episode_actions = []
            episode_rewards = []

            for _ in range(self.args.episode_limit):

                a_n = [agent.choose_action(obs, noise_std=0) for agent, obs in zip(self.agent_n, obs_n)]  # 不添加噪声
                obs_next_n, r_n, done_n, _, _ = self.env_evaluate.step(copy.deepcopy(a_n))
                obs_next_n = self.convert_wrap(obs_next_n)
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

        # 定义目标点和相对位置点
        target = self.env_evaluate.TARGET_POS[0]
        relative = self.env_evaluate.relative_pos + target

        # 绘制目标点
        ax.scatter(target[0], target[1], target[2], color='k', s=100, label='Target')

        # 生成三种颜色映射
        cmaps = [plt.cm.Reds, plt.cm.Greens, plt.cm.Blues]  # 三种渐变色
        norms = [plt.Normalize(min_reward-10, max_reward) for _ in range(3)]

        for agent_id in range(self.args.N_drones):
            agent_states = states[:, agent_id, :3]
            agent_rewards = rewards[:, agent_id]
            cmap = cmaps[agent_id % len(cmaps)]  # 循环使用渐变色
            norm = norms[agent_id % len(norms)]

            # 减少绘制点数
            step_size = max(1, len(agent_states) // 500)  # 绘制最多 100 个点

            # 绘制轨迹，根据奖励值调整颜色亮度
            for i in range(0, len(agent_states) - 1, step_size):
                color_intensity = norm(agent_rewards[i])
                line_color = cmap(color_intensity)
                ax.plot(agent_states[i:i + 2, 0], agent_states[i:i + 2, 1], agent_states[i:i + 2, 2],
                        color=line_color)

            # 绘制相对位置点
            ax.scatter(relative[agent_id, 0], relative[agent_id, 1], relative[agent_id, 2],
                       color=cmap(norm(np.max(agent_rewards))), s=100)

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
    parser.add_argument("--evaluate_freq", type=float, default=100000,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=1, help="Evaluate times")
    parser.add_argument("--max_action", type=float, default=1.0, help="Max action")

    parser.add_argument("--algorithm", type=str, default="MATD3", help="MADDPG or MATD3")
    parser.add_argument("--buffer_size", type=int, default=int(1e6), help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--noise_std_init", type=float, default=0.2, help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_std_min", type=float, default=0.05, help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_decay_steps", type=float, default=3e5,
                        help="How many steps before the noise_std decays to the minimum")
    parser.add_argument("--use_noise_decay", type=bool, default=True, help="Whether to decay the noise_std")
    parser.add_argument("--lr_a", type=float, default=5e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=5e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="Softly update the target network")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Orthogonal initialization")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip")
    parser.add_argument("--device", type=str, default="cpu", help="Device to train model")
    # --------------------------------------MATD3--------------------------------------------------------------------
    parser.add_argument("--policy_noise", type=float, default=0.2, help="Target policy smoothing")
    parser.add_argument("--noise_clip", type=float, default=0.5, help="Clip noise")
    parser.add_argument("--policy_update_freq", type=int, default=2, help="The frequency of policy updates")

    parser.add_argument("--mark", type=int, default=3, help="The frequency of policy updates")
    parser.add_argument("--N_drones", type=int, default=3, help="The number of drones")
    args = parser.parse_args()
    args.noise_std_decay = (args.noise_std_init - args.noise_std_min) / args.noise_decay_steps

    # todo change mark !!!!!!
    runner = Runner(args)
    runner.run()
