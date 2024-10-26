import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import cv2 
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make('CartPole-v1')

GAMMA = 0.99           
LR = 1e-3              
BATCH_SIZE = 64      
MEMORY_SIZE = 10000   
EPSILON_START = 1.0    
EPSILON_END = 0.01     
EPSILON_DECAY = 500   
TARGET_UPDATE = 10     
MAX_EPISODES = 500     

replay_buffer = deque(maxlen=MEMORY_SIZE)

def build_dqn(state_size, action_size):
    return nn.Sequential(
        nn.Linear(state_size, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, action_size)
    ).to(device)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
policy_net = build_dqn(state_size, action_size)
target_net = build_dqn(state_size, action_size)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()  
optimizer = optim.Adam(policy_net.parameters(), lr=LR)

def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()  
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        return policy_net(state).argmax(dim=1).item()  

def store_transition(state, action, reward, next_state, done):
    replay_buffer.append((state, action, reward, next_state, done))

def sample_batch():
    batch = random.sample(replay_buffer, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions).unsqueeze(1).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

    return states, actions, rewards, next_states, dones

def train():
    if len(replay_buffer) < BATCH_SIZE:
        return

    states, actions, rewards, next_states, dones = sample_batch()

    with torch.no_grad():
        max_next_q = target_net(next_states).max(1, keepdim=True)[0]
        target_q = rewards + (1 - dones) * GAMMA * max_next_q

    current_q = policy_net(states).gather(1, actions)

    loss = nn.MSELoss()(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def visualize_cartpole(state):
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    
    cart_width, cart_height = 100, 20
    cart_x = int(state[0] * 100 + 300)  
    cart_y = 350
    cv2.rectangle(img, (cart_x - cart_width // 2, cart_y - cart_height // 2),
                  (cart_x + cart_width // 2, cart_y + cart_height // 2), (0, 0, 255), -1)

    pole_length = 100
    pole_angle = state[2]
    pole_end_x = int(cart_x + pole_length * np.sin(pole_angle))
    pole_end_y = int(cart_y - pole_length * np.cos(pole_angle))
    cv2.line(img, (cart_x, cart_y), (pole_end_x, pole_end_y), (255, 0, 0), 5)

    cv2.line(img, (0, 380), (600, 380), (0, 255, 0), 5)

    cv2.imshow("CartPole", img)
    cv2.waitKey(1)

def train_dqn():
    rewards_per_episode = []  

    plt.ion() 
    fig, ax = plt.subplots()
    ax.set_title("Total Rewards per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.grid()
    
    x = []
    y = []

    for episode in range(MAX_EPISODES):
        state = env.reset()[0]  
        total_reward = 0
        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-episode / EPSILON_DECAY)

        for t in range(200):
            visualize_cartpole(state) 
            action = select_action(state, epsilon) 
            next_state, reward, terminated, truncated, _ = env.step(action) 
            done = terminated or truncated  
            store_transition(state, action, reward, next_state, done) 
            state = next_state
            total_reward += reward

            train() 

            if done:
                break

        rewards_per_episode.append(total_reward)  
        x.append(episode)
        y.append(total_reward)

        ax.clear()
        ax.plot(x, y, color='b')
        ax.set_title("Total Rewards per Episode")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.grid()

        plt.pause(0.01) 

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode}: Total Reward = {total_reward}")

    env.close()
    plt.ioff() 
    plt.show()

if __name__ == "__main__":
    train_dqn()  
    cv2.destroyAllWindows()
