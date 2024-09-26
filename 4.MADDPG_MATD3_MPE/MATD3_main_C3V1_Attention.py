import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
import copy
from utils.replay_buffer import ReplayBuffer
from utils.maddpg import MADDPG
from utils.matd3_attention import MATD3
from gym_pybullet_drones.envs.C3V1 import C3V1
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

Env_name = 'c3v1'  # 'spread3d', 'simple_spread'
action = 'vel'
observation = 'kin_target'  # 相比kin_target 观测会多一个Fs


class Runner:
    def __init__(self, args):
        self.args = args
        self.args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env_name = Env_name
        self.number = args.N_drones
        self.seed = 1145  # 保证一个seed，名称使用记号--mark
        self.mark = args.mark
        self.load_mark = None
        self.args.share_prob = 0.05  # 还是别共享了，有些无用
        Ctrl_Freq = args.Ctrl_Freq  # 30
        self.env = C3V1(gui=True, num_drones=args.N_drones, obs=ObservationType(observation),
                        act=ActionType(action),
                        ctrl_freq=Ctrl_Freq,  # 这个值越大，仿真看起来越慢，应该是由于频率变高，速度调整的更小了
                        need_target=True, obs_with_act=True)
        self.env_evaluate = C3V1(gui=False, num_drones=args.N_drones,
                                 obs=ObservationType(observation),
                                 act=ActionType(action),
                                 ctrl_freq=Ctrl_Freq,
                                 need_target=True, obs_with_act=True)
        self.timestep = 1 / Ctrl_Freq  # 计算每个步骤的时间间隔 0.003

        # self.env.observation_space.shape = box[N,78]
        self.args.obs_dim_n = [self.env.observation_space[i].shape[0] for i in
                               range(self.args.N_drones)]  # obs dimensions of N agents
        self.args.action_dim_n = [self.env.action_space[i].shape[0] for i in
                                  range(self.args.N_drones)]  # actions dimensions of N agents
        # print("observation_space=", self.env.observation_space)
        print(f"obs_dim_n={self.args.obs_dim_n}")
        # print("action_space=", self.env.action_space)
        print(f"action_dim_n={self.args.action_dim_n}")

        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Create N agents
        if self.args.algorithm == "MADDPG":
            print("Algorithm: MADDPG")
            self.agent_n = [MADDPG(self.args, agent_id) for agent_id in range(args.N_drones)]
        elif self.args.algorithm == "MATD3":
            print("Algorithm: MATD3")
            self.agent_n = MATD3.initialize_agents(self.args)
        else:
            print("Wrong!!!")
        self.replay_buffer = ReplayBuffer(self.args)

        # Create a tensorboard
        self.writer = SummaryWriter(
            log_dir=f'runs/{self.args.algorithm}/env_{self.env_name}_number_{self.number}_mark_{self.mark}')
        print(f'存储位置:env_{self.env_name}_number_{self.number}_mark_{self.mark}')
        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0
        self.noise_std = self.args.noise_std_init  # Initialize noise_std

        if self.load_mark is not None:
            for agent_id in range(self.args.N_drones):
                # 加载模型参数
                model_path = "./model/{}/{}_actor_mark_{}_number_{}_step_{}k_agent_{}.pth".format(self.env_name,
                                                                                                  self.args.algorithm,
                                                                                                  self.load_mark,
                                                                                                  self.number,
                                                                                                  int(10000),
                                                                                                  agent_id)  # agent_id
                self.agent_n[agent_id].actor.load_state_dict(torch.load(model_path))

    def run(self, ):
        while self.total_steps < self.args.max_train_steps:
            obs_n, _ = self.env.reset()  # gym new api
            episode_total_reward = 0  # 改进：train_reward -> episode_total_reward，表示当前episode的总奖励
            agent_rewards = [0] * self.args.N_drones  # 改进：rewards_n -> agent_rewards，表示每个智能体的累计奖励

            for count in range(self.args.episode_limit):

                actions_n = [agent.choose_action(obs, noise_std=self.noise_std) for agent, obs in
                             zip(self.agent_n, obs_n)]
                obs_next_n, rewards_n, done_n, _, _ = self.env.step(copy.deepcopy(actions_n))  # gym new api
                # obs_next_n = self.convert_wrap(obs_next_n)

                self.replay_buffer.store_transition(obs_n, actions_n, rewards_n, obs_next_n, done_n)
                obs_n = obs_next_n
                episode_total_reward += np.mean(rewards_n)  # 当前episode的总奖励
                agent_rewards = [cumulative_reward + reward for cumulative_reward, reward in
                                 zip(agent_rewards, rewards_n)]  # 每个智能体的累计奖励
                self.total_steps += 1

                if self.args.use_noise_decay:
                    self.noise_std = self.noise_std - self.args.noise_std_decay if self.noise_std - self.args.noise_std_decay > self.args.noise_std_min else self.args.noise_std_min

                if self.total_steps % self.args.evaluate_freq == 0:
                    # self.evaluate_policy()
                    self.save_model()  # 评估中实现save了
                    obs_n, _ = self.env.reset()  # gym new api

                if all(done_n):
                    break

            if self.replay_buffer.current_size > self.args.batch_size:
                for _ in range(50):
                    for agent_id in range(self.args.N_drones):
                        self.agent_n[agent_id].train(self.replay_buffer, self.agent_n)

            print(f"total_steps:{self.total_steps} \t episode_total_reward:{int(episode_total_reward)} \t "
                  f"noise_std:{self.noise_std}")

            for agent_id, cumulative_reward in enumerate(agent_rewards):
                self.writer.add_scalar(f'Agent_{agent_id}_train_reward', int(cumulative_reward),
                                       global_step=self.total_steps)

            self.writer.add_scalar(f'train_step_rewards_{self.env_name}', int(episode_total_reward),
                                   global_step=self.total_steps)

        self.env.close()
        self.env_evaluate.close()

    def evaluate_policy(self, ):
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            obs_n, _ = self.env_evaluate.reset()
            episode_reward = 0
            a = self.agent_n[0].choose_action(obs_n[0], 0)
            for _ in range(self.args.test_episode_limit):

                a_n = [agent.choose_action(obs, noise_std=0) for agent, obs in
                       zip(self.agent_n, obs_n)]  # We do not add noise when evaluating
                obs_next_n, r_n, done_n, _, _ = self.env_evaluate.step(copy.deepcopy(a_n))
                episode_reward += np.mean(r_n)  # 修改为均值
                obs_n = obs_next_n

                if all(done_n):
                    break
            evaluate_reward += episode_reward

        evaluate_reward = evaluate_reward / self.args.evaluate_times
        self.evaluate_rewards.append(evaluate_reward)
        # print("total_steps:{} \t evaluate_reward:{} \t noise_std:{}".format(self.total_steps, evaluate_reward,
        #                                                                     self.noise_std))
        # self.writer.add_scalar('evaluate_step_rewards_{}'.format(self.env_name), evaluate_reward,
        #                        global_step=self.total_steps)
        # Save the rewards and models
        # np.save('./data_train/{}_env_{}_number_{}_seed_{}.npy'.format(self.args.algorithm, self.env_name, self.number,
        #                                                               self.seed), np.array(self.evaluate_rewards))
        for agent_id in range(self.args.N_drones):
            self.agent_n[agent_id].save_model(self.env_name, self.args.algorithm, self.mark, self.number,
                                              self.total_steps,
                                              agent_id)

    def save_model(self):
        for agent_id in range(self.args.N_drones):
            self.agent_n[agent_id].save_model(self.env_name, self.args.algorithm, self.mark, self.number,
                                              self.total_steps,
                                              agent_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MADDPG and MATD3 in MPE environment")
    parser.add_argument("--max_train_steps", type=int, default=int(1e6), help=" Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=1500, help="Maximum number of steps per episode")
    parser.add_argument("--test_episode_limit", type=int, default=1500, help="Maximum number of steps per test episode")
    parser.add_argument("--evaluate_freq", type=float, default=100000,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=1, help="Evaluate times")
    parser.add_argument("--max_action", type=float, default=1.0, help="Max action")

    parser.add_argument("--algorithm", type=str, default="MATD3", help="MADDPG or MATD3")
    parser.add_argument("--buffer_size", type=int, default=int(1e6), help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")  # 1024-》4048
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--noise_std_init", type=float, default=0.05, help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_std_min", type=float, default=0, help="The std of Gaussian noise for exploration")
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

    parser.add_argument("--mark", type=int, default=1145, help="The frequency of policy updates")
    parser.add_argument("--N_drones", type=int, default=3, help="The number of drones")
    parser.add_argument("--Ctrl_Freq", type=int, default=30, help="The frequency of ctrl")
    args = parser.parse_args()
    args.noise_std_decay = (args.noise_std_init - args.noise_std_min) / args.noise_decay_steps

    runner = Runner(args)
    runner.run()
