import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class BlackHoleEnv(gym.Env):
    def __init__(self):
        super(BlackHoleEnv, self).__init__()
        self.position = np.array([1,1], dtype=np.float64)
        self.dt = 0.000001  # 시간 단위
        self.current_step = 0
        self.c = 3*10**8  # 광속 (m/s)
        self.G = 6.7*10**-11  # 중력 상수 (m^3 kg^-1 s^-2 d )
        self.black_hole_mass = 2*10**30  # 블랙홀 질량
        self.success1 = False
        self.success2 = False
        self.fall1 = False
        self.fall2 = False
        self.rewards = -100
        self.trajectory = []

        self.action_space = spaces.Box(low=4001, high=8000, shape=(1,), dtype=np.float64)
        self.observation_space = spaces.MultiBinary(4)

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None: 
            self.seed(seed)
        self.position = np.array([1, 1], dtype=np.float64)
        self.velocity = np.array([1, 1], dtype=np.float64)
        self.done = False
        self.current_step = 0
        self.trajectory=[]
        self.rewards = 0
        return self._get_obs()

    def _get_obs(self):
        return np.array([self.success1, self.success2, self.fall1, self.fall2], dtype=np.bool_)

    def step(self, action):
        self.position = np.array([0, action], dtype=np.float64)
        self.terminated1 = False
        self.truncated1 = False
        self.terminated2 = False
        self.truncated2 = False
        self.success1 = False
        self.success2 = False
        self.fall1 = False
        self.fall2 = False
        terminated = False
        truncated = False
        self.trajectory.append(self.position.copy())
        for i in range(10000):
            self.gravity = self.G * self.black_hole_mass/np.linalg.norm(self.position)**2
            self.acc = 9*10**16 /np.linalg.norm(self.position)
            self.current_step += 1
            unit_vector = -self.position / np.linalg.norm(self.position)
            self.velocity += self.gravity * unit_vector * self.dt
            # 시계 반대 방향으로의 접선 벡터 계산
            tangent_vector = np.array([-self.position[1], self.position[0]])


            # action 벡터를 접선 방향으로 설정
            self.velocity += self.acc * tangent_vector/np.linalg.norm(tangent_vector) * self.dt
            self.position += self.velocity * self.dt
            self.trajectory.append(self.position.copy())

            if np.linalg.norm(self.position) <= 3500:  # 에르고 영역 진입
                if np.linalg.norm(self.position) <= 3000:  # 블랙홀 중심에 빨려 들어가는 경우
                    self.rewards += -800
                    self.terminated1 = True
                    self.fall1 = True
                    break
                else:
                    self.rewards += 3000
                    self.terminated1 = True
                    self.success1 = True
                    break
            elif np.linalg.norm(self.position) >= 100000:
                self.rewards += - 600
                self.terminated1 = True
                break
            elif i >=9999:
                self.rewards += -300
                self.truncated1 = True
                break
            else:
                self.rewards += -10  # 탈출을 촉진하기 위해 각 단계에 작은 벌점
    
        for i in range(10000):
            self.gravity = self.G * self.black_hole_mass/np.linalg.norm(self.position)**2
            self.acc = 9*10**16 /np.linalg.norm(self.position)
            self.current_step += 1
            unit_vector = -self.position / np.linalg.norm(self.position)
            self.velocity += self.gravity * unit_vector * self.dt
            # 시계 반대 방향으로의 접선 벡터 계산
            tangent_vector = np.array([-self.position[1], self.position[0]])
            tangent_vector = tangent_vector / np.linalg.norm(tangent_vector)  # 단위 벡터로 정규화

            # action 벡터를 접선 방향으로 설정
            self.velocity += self.acc * tangent_vector * self.dt
            self.position += self.velocity * self.dt
            self.trajectory.append(self.position.copy())

            if np.linalg.norm(self.position) >= 4500:  # 블랙홀 탈출
                self.rewards += 3150
                self.terminated2 = True
                self.success2 = True
                break
            elif np.linalg.norm(self.position) <= 3000:  # 블랙홀 중심에 빨려 들어가는 경우
                self.rewards += -700
                self.terminated2 = True
                self.fall2 = True
                break
            elif i >=9999:
                self.rewards += -300
                self.truncated2 = True
                break
            else:
                self.rewards += -10  # 탈출을 촉진하기 위해 각 단계에 작은 벌점
        if self.truncated1 or self.truncated2:
            truncated=True
        else:
            terminated=True

        return self._get_obs(), self.rewards, terminated, truncated, {}
    

env = BlackHoleEnv()

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
    
    def push(self, transition):
        self.memory.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

def train(env, agent, optimizer, criterion, memory, batch_size, gamma):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    batch = list(zip(*transitions))

    states = torch.tensor(np.array(batch[0]), dtype=torch.float32)
    actions = torch.tensor(np.array(batch[1]), dtype=torch.int64).unsqueeze(1)
    rewards = torch.tensor(np.array(batch[2]), dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(np.array(batch[3]), dtype=torch.float32)
    dones = torch.tensor(np.array(batch[4]), dtype=torch.float32).unsqueeze(1)

    q_values = agent(states).gather(1, actions)
    next_q_values = agent(next_states).max(1)[0].unsqueeze(1)
    target_q_values = rewards + gamma * next_q_values * (1 - dones)

    loss = criterion(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = DQN(state_dim, action_dim)
optimizer = optim.Adam(agent.parameters(), lr=0.001)
criterion = nn.MSELoss()
memory = ReplayMemory(10000)
batch_size = 64
gamma = 0.99
num_episodes = 100
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 0.995

# Training Loop
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        if random.random() < epsilon:
            action = np.random.uniform(4001, 8000)
        else:
            q_values = agent(state_tensor)
            action = q_values.argmax().item()
        print('action for this ep:', action)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        memory.push((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        train(env, agent, optimizer, criterion, memory, batch_size, gamma)
    print(f"Episode {episode + 1}, Total Reward: {total_reward}")



# Evaluation
def evaluate(env, agent, num_episodes=1):
    total_rewards = 0
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = agent(state_tensor)
            action = q_values.argmax().item()
            print('action for this ep:', action)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward
        total_rewards += episode_reward
        print(f"Test Episode {episode + 1}, Reward: {episode_reward}")
    avg_reward = total_rewards / num_episodes
    print(f"Average Reward over {num_episodes} Test Episodes: {avg_reward}")

evaluate(env, agent)

# Plot the trajectory
trajectory1 = np.array(env.trajectory)  # Convert to numpy array for plotting
plt.plot(trajectory1[:, 0], trajectory1[:, 1], marker='o')
plt.scatter(0, 0, color='red', label='Black Hole Center')
plt.xlim(-100000, 100000)
plt.ylim(-100000, 100000)
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.show()

# Print trajectory coordinates and results
print("Trajectory coordinates:", trajectory1)