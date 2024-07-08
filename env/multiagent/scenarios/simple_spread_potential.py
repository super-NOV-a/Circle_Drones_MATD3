import numpy as np
import torch

from env.multiagent.core import World, Agent, Landmark
from env.multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        num_landmarks = 3
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
            agent.observation_field = 1
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists)
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    def observation(self, agent, world):
        num_landmark = 3
        num_agent = 3
        # 根据视野范围内的landmark计算势能
        # 对于其他agent，记录当前的绝对位置和速度
        other_agent_pos = []
        other_agent_vel = []
        visible_landmark_pos = []
        for other in world.agents:
            if other is agent:
                continue
            other_agent_pos.append(other.state.p_pos)
            other_agent_vel.append(other.state.p_vel)

        landmark_idx = []
        occupy_idx_matrix = []
        for i, entity in enumerate(world.landmarks):
            dist = np.sqrt(np.sum(np.square(agent.state.p_pos - entity.state.p_pos)))
            # 如果可以看见某个landmark，记录它的位置，记录它的索引
            if dist <= agent.observation_field:
                visible_landmark_pos.append(entity.state.p_pos)
                landmark_idx.append(i)
                # 如果和某个landmark的距离过近，landmark被占领，占领者为当前agent
                if dist <= world.occupy_size:
                    entity.occupy = True
                    entity.occupy_idx = int(agent.name[-1])
                    occupy_idx_matrix.append(entity.occupy_idx)
                    continue
                occupy_idx_matrix.append(entity.occupy_idx)
            # 如果看不见，idx里标记为-1
            else:
                visible_landmark_pos.append(np.zeros(entity.state.p_pos.size))
                landmark_idx.append(-1)
                occupy_idx_matrix.append(entity.occupy_idx)

        occupy_matrix = []
        for i, entity in enumerate(world.landmarks):
            # 如果被占领，则置0，没被占领置1
            if entity.occupy:
                occupy_matrix.append(0)
            else:
                occupy_matrix.append(1)

        # obs = 2+2+2*2+3*2 = 14
        obs = np.concatenate([agent.state.p_pos] + [agent.state.p_vel] + other_agent_pos + visible_landmark_pos)
        # 结合landmark占领情况,计算当前的势能
        F = potential_energy(obs, num_landmark, num_agent, world.dim_p, occupy_matrix)
        return np.concatenate((obs,F))


# https://blog.csdn.net/weixin_42301220/article/details/125155505
def potential_energy(obs, num_landmark, num_agent, dim_p, occupy_matrix):
    num_other_agent = num_agent-1
    # 自身位置 obs(0,2)
    self_pos = obs[0:dim_p]
    #print(self_pos)
    # 引力增益系数，agent和landmark之间离得越近越大
    eta_att = 10
    # 斥力增益系数,agent之间斥力离得越近越大
    eta_rep_agent = 2
    # 斥力感应范围
    d0 = 1
    # 调节因子
    n =1
    # 储存受到的agent斥力
    F_rep_agent = []
    # 储存受到的引力
    F_att = []
    # 储存other agent坐标 obs(4,8) 2个
    agents_pos = []
    for i in range(dim_p*2, dim_p*2+num_other_agent*dim_p, dim_p):
        agents_pos.append(obs[i:i+dim_p])
    #print(agents_pos)
    # 储存landmark坐标 obs(8,14) 3个
    landmarks_pos = []
    for i in range((2+num_other_agent)*dim_p, (2+num_other_agent+num_landmark)*dim_p, dim_p):
        landmarks_pos.append(obs[i:i+dim_p])
    #print(landmarks_pos)

    # 保存agent当前位置和其他agent的方向向量
    delta_ag = []
    dists_ag = []
    unit_ag = []
    # 计算agent当前和每个其他agent之间的单位方向向量
    for i in range(num_agent-1):
        delta_ag.append(self_pos - agents_pos[i])
        dists_ag.append(np.linalg.norm(delta_ag[i]))
        unit_ag.append(delta_ag[i]/dists_ag[i])

    # 保存agent当前位置和landmark之间的方向向量，如果某个landmark不可见就不计算
    # 从agent指向landmark，引力，大于某个范围就没有引力或者斥力了
    delta_lm = []
    dists_lm = []
    unit_lm = []
    for i in range(num_landmark):
        if np.all(landmarks_pos[i] == 0.0):
            delta_lm.append(np.array((0.0, 0.0)))
            dists_lm.append(np.array((0.0, 0.0)))
            unit_lm.append(np.array((0.0, 0.0)))
        else:
            delta_lm.append(landmarks_pos[i] - self_pos)
            dists_lm.append(np.linalg.norm(delta_lm[i]))
            unit_lm.append(delta_lm[i]/dists_lm[i])

    # 计算当前agent和每个landmark之间的引力，输出为一个list
    #F_att = eta_att*[a * b for a,b in zip(dists_lm, unit_lm)]
    for i in range(num_landmark):
        # 如果某个landmark看不见，她的引力为0
        if np.all(dists_lm[i] == 0.0):
            F_att.append(np.array((0.0, 0.0)))
        else:
            # 计算大小
            F_att_abs = eta_att*(1/dists_lm[i])/dists_lm[i]**2
            # 大小×方向则为这个landmark给的吸引力，存入F_att
            F_att.append(unit_lm[i] * F_att_abs)
    # 被占领的landmark无法对agent产生引力
    F_att_enable = [a * b for a, b in zip(F_att, occupy_matrix)]
    F_att_mean = np.mean(F_att_enable, axis=0)
    #print("引力：", F_att_mean)

    # 计算agent间斥力,分别计算，有两个向量
    for i in range(num_agent-1):
        # 距离过远没有斥力
        if dists_ag[i] >= d0:
            F_rep_agent.append(np.array((0.0, 0.0)))
        else:
            # agent的斥力1，方向由另外agent指向本agent
            # 斥力大小，计算得到agent之间的，受可观察landmark影响
            # 每个landmark影响算出一个大小，取均值
            F_rep_ob1_abs = [eta_rep_agent*(1/dists_ag[i]-1/d0)/dists_ag[i]**2 + eta_rep_agent*(1/dists_ag[i] - 1/d0)*(dists_lm[j])**n/dists_ag[i]**2 for j in range(num_landmark)]
            # 斥力方向 list:3,每个landmark对斥力大小的影响不同，但方向一致
            F_rep_ob1 = [unit_ag[i]*a for a in F_rep_ob1_abs]
            F_rep_ob1_mean = np.mean(F_rep_ob1, axis=0)
            # agent的斥力2，方向由agent指向landmark
            F_rep_ob2_abs = []
            for j in range(num_landmark):
                if np.all(dists_lm[j] == 0.0):  # 看不见某个landmark，则无斥力2
                    F_rep_ob2_abs.append(np.array((0.0, 0.0)))
                else:
                    F_rep_ob2_abs.append(n / 2 * eta_rep_agent * (1 / dists_ag[i] - 1 / d0) ** 2 / ((dists_lm[j]) ** n))
            F_rep_ob2 = [a * b for a, b in zip(F_rep_ob2_abs, unit_lm)]
            # 被占领的landmark不产生斥力2
            F_rep_ob2_enable = [a * b for a, b in zip(F_rep_ob2_abs, occupy_matrix)]
            F_rep_ob2_mean = np.mean(F_rep_ob2_enable, axis=0)
            # 计算合斥力
            F_rep_agent.append(F_rep_ob1_mean+F_rep_ob2_mean)
    F_rep = np.mean(F_rep_agent,axis=0)
    #print("斥力：", F_rep)

    F = F_att_mean + F_rep

    #print("合力：", F)
    # 合力等比例缩小到0-1范畴间
    # F = scaled_potential(F, 10)
    # 对合力进行缩放
    #F = torch.softmax(torch.tensor(F),1)
    return F