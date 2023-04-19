import gym
import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)



def epsilon_greedy_policy(q_values, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(q_values.shape[-1])
    else:
        return torch.argmax(q_values).item()

def optimize_model(optimizer, loss_fn, batch, dqn, target_dqn, gamma):
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(-1)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.bool)

    q_values = dqn(states).gather(1, actions).squeeze()
    with torch.no_grad():
        next_q_values = target_dqn(next_states).max(1).values
        target_q_values = rewards + gamma * next_q_values * (~dones)

    loss = loss_fn(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def epsilon_greedy_policy(q_values, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(q_values.shape[-1])
    else:
        return torch.argmax(q_values).item()

def optimize_model(optimizer, loss_fn, batch, dqn, target_dqn, gamma):
    states, actions, rewards, next_states, dones = zip(*batch)
    states= np.array(states)
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(-1)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states= np.array(next_states)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones= np.array(dones)
    dones = torch.tensor(dones, dtype=torch.bool)

    q_values = dqn(states).gather(1, actions).squeeze()
    with torch.no_grad():
        next_q_values = target_dqn(next_states).max(1).values
        target_q_values = rewards + gamma * next_q_values * (~dones)

    loss = loss_fn(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.model(x)

def test_agent(env, dqn):
    state = env.reset()
    done = False
    episode_reward = 0


    while not done:
        #env.render()
        q_values = dqn(torch.tensor(state, dtype=torch.float32))
        action = torch.argmax(q_values).item()
        state, reward, done, _ = env.step(action)
        episode_reward += reward

    return episode_reward, state[0] >= env.goal_position





if __name__== "__main__":
    # Initialize the environment and DQN agent
    env = gym.make('MountainCar-v0')
    #print(env.goal_position)
    env.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    dqn = DQN(state_dim, action_dim)
    target_dqn = DQN(state_dim, action_dim)
    target_dqn.load_state_dict(dqn.state_dict())

    optimizer = optim.Adam(dqn.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    memory = deque(maxlen=10000)
    batch_size = 64
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 0.995
    num_episodes = 1000

    epsilon = epsilon_start
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            q_values = dqn(torch.tensor(state, dtype=torch.float32))
            action = epsilon_greedy_policy(q_values, epsilon)
            next_state, reward, done, _ = env.step(action)
            memory.append((state, action, reward, next_state, done))

            state = next_state
            episode_reward += reward

            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                loss = optimize_model(optimizer, loss_fn, batch, dqn, target_dqn, gamma)

            if done:
                episode_rewards.append(episode_reward)
                print(f"Episode: {episode}, Reward: {episode_reward}, Epsilon: {epsilon}")

                if episode % 10 == 0:
                    target_dqn.load_state_dict(dqn.state_dict())

                epsilon = max(epsilon_final, epsilon * epsilon_decay)

    env.close()

    test_episodes = 1000
    test_rewards = []

    for _ in range(test_episodes):
        test_reward, pos = test_agent(env, dqn)
        if test_reward> -200 or pos:
            print(test_reward)
            test_rewards.append(test_reward)

    env.close()
    print(f"Mean test reward: {np.mean(test_rewards)}")
