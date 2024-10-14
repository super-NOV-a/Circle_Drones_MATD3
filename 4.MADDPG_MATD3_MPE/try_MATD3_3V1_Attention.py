import os
import random

import torch
import numpy as np
import argparse
from utils.matd3_attention import MATD3
import copy
from gym_pybullet_drones.envs.C3V1_Test import C3V1_Test
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
import matplotlib.pyplot as plt
import plotly.graph_objects as go


Env_name = 'c3v1A'  # c3v1 \ c3v1A (最好的为9200)\ c3v1G\c3v1A_GR (最好的为9200)
Mark = 9200  # todo 测试时指定mark
action = 'vel'
Eval_plot = True                # 是否绘制轨迹图像 该选项同时保存txt和png文件,重复保存会覆盖 该选项会增加运行时间!
Env_gui = False                 # 环境gui是否开启 建议关闭 该选项会增加时间
Display = False                 # 开启Eval_plot后：绘制图像是否展示 建议关闭，想看去文件夹下面看去
Need_Html = False               # 开启Eval_plot后：是否需要Html图像 建议关闭
Success_Time_Limit = 1000       # 成功时间限制，max: 1000, 不在环境中定义 todo 修改成功条件
Success_FollowDistance = 1.0      # 成功靠近目标距离: 1。跟踪敌机的距离 胜利条件
Success_AttackDistance = 0.13    # 成功打击距离: 0.1。打击敌机的距离 胜利条件
Success_KeepDistance = 0.1      # 彼此不碰撞距离: 0.1。不碰撞距离 成功条件
# 胜利条件=时间限制+跟踪敌机的距离限制+打击敌机的距离限制
# 成功条件(完美条件)=时间限制+跟踪敌机的距离限制+打击敌机的距离限制+彼此不碰撞距离条件


class Runner:
    def __init__(self, args):
        self.args = args
        self.args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env_name = Env_name
        self.number = 3  #
        self.seed = 1145  # 保证一个seed，名称使用记号--mark
        self.mark = Mark  # todo 指定mark
        Load_Steps = 10000000  # self.args.max_train_steps = 1e6
        self.test_times = 300  # 修改为100次运行
        self.done_count = 0  # 用于记录胜利次数
        self.success_count = 0  # 用于记录成功次数（完美条件）
        # Create env
        self.env_evaluate = C3V1_Test(gui=Env_gui, num_drones=args.N_drones, obs=ObservationType('kin_target'),
                                      act=ActionType(action),
                                      ctrl_freq=30,  # 这个值越大，仿真看起来越慢，应该是由于频率变高，速度调整的更小了
                                      need_target=True, obs_with_act=True,
                                      follow_distance=Success_FollowDistance,
                                      acctack_distance=Success_AttackDistance,
                                      keep_distance=Success_KeepDistance)
        self.timestep = 1.0 / 30  # 计算每个步骤的时间间隔 0.003

        self.args.obs_dim_n = [self.env_evaluate.observation_space[i].shape[0] for i in
                               range(self.args.N_drones)]  # obs dimensions of N agents
        self.args.action_dim_n = [self.env_evaluate.action_space[i].shape[0] for i in
                                  range(self.args.N_drones)]  # actions dimensions of N agents
        print("obs_dim_n={}".format(self.args.obs_dim_n))
        print("action_dim_n={}".format(self.args.action_dim_n))

        # Set random seed
        self.set_random_seed(self.seed)

        # Create N agents
        if self.args.algorithm == "MATD3":
            print("Algorithm: MATD3")
            self.agent_n = [MATD3(self.args, agent_id) for agent_id in range(args.N_drones)]
        else:
            print("Wrong algorithm!!!")
        # 加载模型参数
        for agent_id in range(self.args.N_drones):
            model_path = "./model/{}/{}_actor_mark_{}_number_{}_step_{}k_agent_{}.pth".format(self.env_name,
                                                                                              self.args.algorithm,
                                                                                              self.mark, self.number,
                                                                                              int(Load_Steps / 1000),
                                                                                              agent_id)  # agent_id
            self.agent_n[agent_id].actor.load_state_dict(torch.load(model_path))
        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.noise_std = self.args.noise_std_init  # Initialize noise_std

    def set_random_seed(self, seed):
        """
        设置固定的随机种子以确保可复现性
        """
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def run(self, ):
        for eval_time in range(self.test_times):
            job_done, success = self.evaluate_policy(Eval_plot, eval_time)
            if job_done:
                self.done_count += 1
                if success:
                    self.success_count += 1
        self.env_evaluate.close()

        # 计算成功率
        done_rate = self.done_count / self.test_times
        success_rate = self.success_count / self.test_times
        print(f"条件靠近距离{Success_FollowDistance},打击距离{Success_AttackDistance},"
              f"任务胜率: {done_rate * 100}%, 完美成功率: {success_rate * 100}%")

    def evaluate_policy(self, eval_plot, eval_time):  # 仅测试一次的
        all_states, all_actions, all_rewards, all_target_pos = [], [], [], []
        Job_done = False  # 用于记录本次运行是否成功
        Success = False
        obs_n, _ = self.env_evaluate.reset()
        self.env_evaluate.collision = False
        episode_return = [0 for _ in range(self.args.N_drones)]
        episode_states = []
        # episode_actions = []
        episode_rewards = []
        episode_target_pos = []

        for episode_len in range(self.args.episode_limit):
            a_n = [agent.choose_action(obs, noise_std=0.005) for agent, obs in zip(self.agent_n, obs_n)]  # 不添加噪声
            # time.sleep(0.01)
            obs_next_n, r_n, done_n, collided, _ = self.env_evaluate.step(copy.deepcopy(a_n))
            for i in range(self.args.N_drones):
                episode_return[i] += r_n[i]

            # 保存状态、动作和奖励
            episode_target_pos.append(self.env_evaluate.TARGET_POS)
            episode_states.append(obs_n)
            # episode_actions.append(a_n)
            episode_rewards.append(r_n)

            obs_n = obs_next_n
            if any(done_n):  # 如果有一个 done 为 True，则算作成功
                Job_done = True  # 打击成功
                if collided is False:  # 期间没有发生碰撞
                    Success = True
                break
            # if episode_len > Success_Time_Limit:
            #     pass

        all_target_pos.append(episode_target_pos)
        all_states.append(episode_states)
        # all_actions.append(episode_actions)
        all_rewards.append(episode_rewards)

        print("打击胜利:{}, 成功：{} \t episode_reward:{} \t".format(Job_done, Success, episode_return))

        # 将数据转换为numpy数组
        if eval_plot:
            all_target_pos = np.array(all_target_pos)
            all_states = np.array(all_states)
            # all_actions = np.array(all_actions)
            all_rewards = np.array(all_rewards)

            # 绘制图
            for a_time in range(self.args.evaluate_times):   # 其实每次 eval 只绘制一次
                self.plot_and_save_results(eval_time, all_states[a_time], all_rewards[a_time],
                                           all_target_pos[a_time], Job_done, Success, Need_Html)

        return Job_done, Success

    def plot_and_save_results(self, time_i, states, rewards, target_pos, Job_done, Success, create_html=False):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        # 计算奖励范围以用于颜色映射
        min_reward = np.min(rewards)
        max_reward = np.max(rewards)
        # 生成三种颜色映射
        cmaps = [plt.cm.Reds, plt.cm.Greens, plt.cm.Blues]  # 三种渐变色
        norms = [plt.Normalize(min_reward - 2, max_reward) for _ in range(3)]
        # 保存路径数据的目录
        save_dir = f"./agent_paths/{self.env_name}_{self.mark}"
        os.makedirs(save_dir, exist_ok=True)
        save_file_path = os.path.join(save_dir, f"{time_i}_胜{Job_done}_成{Success}.txt")
        # 定义不同的颜色，用于区分目标点和无人机
        colors = ['red', 'green', 'blue', 'yellow']     # 三个无人机和一个目标位置
        with open(save_file_path, 'w') as f:
            # 保存目标点数据
            f.write("# Target Trajectories (x, y, z)\n")
            for i in range(len(target_pos)):
                f.write(f"{target_pos[i, 0]}, {target_pos[i, 1]}, {target_pos[i, 2]}\n")

            # 绘制目标点轨迹（目标位置曲线）
            ax.plot(target_pos[:, 0], target_pos[:, 1], target_pos[:, 2],
                    color=colors[3], label='Target Position', linestyle='--')

            # 保存智能体轨迹数据
            f.write("\n# Agent Trajectories (x, y, z)\n")
            for agent_id in range(self.args.N_drones):
                agent_states = states[:, agent_id, :3]  # 提取无人机位置信息
                agent_rewards = rewards[:, agent_id]
                cmap = cmaps[agent_id % len(cmaps)]  # 循环使用渐变色
                norm = norms[agent_id % len(norms)]

                f.write(f"\nAgent {agent_id} trajectory:\n")
                for i in range(len(agent_states)):
                    f.write(f"{agent_states[i, 0]}, {agent_states[i, 1]}, {agent_states[i, 2]}\n")

                # 减少绘制点数
                step_size = max(1, len(agent_states) // 500)  # 绘制最多 500 个点

                # 绘制无人机轨迹，根据奖励值调整颜色亮度
                for i in range(0, len(agent_states) - 1, step_size):
                    color_intensity = norm(agent_rewards[i])
                    line_color = cmap(color_intensity)
                    ax.plot(agent_states[i:i + 2, 0], agent_states[i:i + 2, 1], agent_states[i:i + 2, 2],
                            color=line_color, label=f'Agent {agent_id}' if i == 0 else "")  # 仅为首次绘制添加标签

                # 突出显示每个无人机轨迹的起始点，不添加标签
                ax.scatter(agent_states[0, 0], agent_states[0, 1], agent_states[0, 2],
                           color=colors[agent_id], s=100, marker='o')  # 起始点使用大号标记符号

        # 设置标题和轴标签
        ax.set_title('Agent and Target Positions Over Time')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        # 调整布局并显示图像
        plt.tight_layout()
        # 保存静态图像
        plt.savefig(os.path.join(save_dir, f"{time_i}_胜{Job_done}_成{Success}.png"))
        # print(f'已经保存:{self.env_name}_{self.mark}_{time_i}相关文件')

        # 判断是否创建HTML
        if create_html:
            # 使用 Plotly 重新绘制并保存交互式 HTML 文件
            fig_html = go.Figure()

            # 添加目标点轨迹
            fig_html.add_trace(go.Scatter3d(x=target_pos[:, 0], y=target_pos[:, 1], z=target_pos[:, 2],
                                            mode='lines', line=dict(color='yellow', dash='dash'),
                                            name='Target Position'))

            # 添加无人机轨迹
            for agent_id in range(self.args.N_drones):
                agent_states = states[:, agent_id, :3]
                fig_html.add_trace(go.Scatter3d(x=agent_states[:, 0], y=agent_states[:, 1], z=agent_states[:, 2],
                                                mode='lines+markers', line=dict(color=colors[agent_id]),
                                                name=f'Agent {agent_id}'))

            # 设置标题和轴标签
            fig_html.update_layout(title='Agent and Target Positions Over Time',
                                   scene=dict(
                                       xaxis_title='X',
                                       yaxis_title='Y',
                                       zaxis_title='Z'),
                                   width=700,
                                   margin=dict(r=20, b=10, l=10, t=10))

            # 保存为HTML
            html_path = os.path.join(save_dir, f"{time_i}_胜{Job_done}_成{Success}.html")
            fig_html.write_html(html_path)
            print(f"HTML saved to {html_path}")

        if Display:
            plt.show()  # 显示静态图像
        else:
            plt.close()


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
