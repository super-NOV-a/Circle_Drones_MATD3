import numpy as np
from env.multiagent.core import World, Agent, Landmark
from env.multiagent.scenario import BaseScenario


# agent有观察范围，超过范围的landmark看不到
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
            agent.observation_field = 0.75   # agent的视野范围
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
        for i, agent in enumerate(world.agents): #初始化agent的颜色
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks): #初始化标志颜色
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        # 初始化agent和标志状态包括位置速度通信等
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
            # 如果每个agent都看不到这个agent，直接惩罚1（在范围内最少-0.75）
            if all(_ > agent.observation_field for _ in dists):
                rew -= 1
            else:
                rew -= min(dists)
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:得到每个标志实体的位置和agent位置差
            dist = np.sqrt(np.sum(np.square(agent.state.p_pos - entity.state.p_pos)))
            # 如果这个landmark可以被看见
            if dist <= agent.observation_field:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            else: # 如果看不见这个landmark
                entity_pos.append(np.zeros_like(entity.state.p_pos))
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:得到每个标志的颜色
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:#对世界中的agent来说，如果送入的agent和other一致，continue，不一致就要交流
            if other is agent: continue
            # state.c写入的是别人的动作，能看见其他智能体的动作
            comm.append(other.state.c) #把除本身外所有agent的状态中的c拿出来放到comm里
            other_pos.append(other.state.p_pos - agent.state.p_pos) #把除本身外所有agent和本身的位置差算一下放进other_pos里
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)#返回2vel+2pos+2*3pos+2*2pos+2*2comm
