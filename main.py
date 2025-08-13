# main.py

import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# --- 1. Imports and Environment Setup ---
# Setup the environment, device for computation, and plotting.

env = gym.make("CartPole-v1")
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# --- 2. Data Structures ---
# Define the Replay Memory for storing and sampling experiences.

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        # A deque is used for efficient memory management.
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        # Saves a new transition.
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        # Randomly samples a batch of transitions.
        return random.sample(self.memory, batch_size)

    def __len__(self):
        # Returns the current size of the memory.
        return len(self.memory)

# --- 3. Agent Architecture (DQN) ---
# Define the neural network that approximates the Q-function.

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        # A simple feed-forward network with two hidden layers.
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        # The forward pass defines how the input data flows through the network.
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# --- 4. Hyperparameters & Initialization ---
# Define key training parameters and initialize the networks, optimizer, and memory.

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)
steps_done = 0
episode_durations = []

# --- 5. Training & Evaluation Functions ---

def select_action(state):
    # Implements the epsilon-greedy policy to balance exploration and exploitation.
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if random.random() > eps_threshold:
        with torch.no_grad():
            return policy_net(state).argmax(1).view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def plot_durations(show_result=False):
    # Plots the duration of episodes and the 100-episode moving average.
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())

    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize_model():
    # Performs a single optimization step on the policy network using the Bellman equation.
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def evaluate_agent(env_eval, policy_net_eval, num_episodes=10, render=False):
    # Evaluates the agent's performance without exploration.
    print("\n--- Starting Evaluation ---")
    policy_net_eval.eval()
    episode_rewards = []
    
    for i_episode in range(num_episodes):
        state_eval, info_eval = env_eval.reset()
        state_eval = torch.tensor(state_eval, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0
        
        for _ in count():
            if render:
                env_eval.render()
            
            with torch.no_grad():
                action_eval = policy_net_eval(state_eval).argmax(1).view(1, 1)
            
            observation_eval, reward_eval, terminated_eval, truncated_eval, _ = env_eval.step(action_eval.item())
            total_reward += reward_eval
            
            if terminated_eval or truncated_eval:
                break
                
            next_state_eval = torch.tensor(observation_eval, dtype=torch.float32, device=device).unsqueeze(0)
            state_eval = next_state_eval
            
        episode_rewards.append(total_reward)
        print(f"Evaluation Episode {i_episode+1}: Total Reward = {total_reward}")
    
    env_eval.close()
    print(f"Average reward over {num_episodes} evaluation episodes: {sum(episode_rewards) / num_episodes:.2f}")
    return episode_rewards

# --- 6. Main Training Loop & Model Saving ---
if __name__ == "__main__":
    NUM_EPISODES = 600
    print(f"Training on {device} for {NUM_EPISODES} episodes...")
    for i_episode in range(NUM_EPISODES):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        for t in count():
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            
            if terminated or truncated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                
            memory.push(state, action, next_state, reward)
            state = next_state
            optimize_model()
            
            # Soft update the target network.
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)
            
            if terminated or truncated:
                episode_durations.append(t + 1)
                plot_durations()
                break
    
    print('Training Complete!')
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()

    # Save the trained model to a 'models' directory.
    os.makedirs('models', exist_ok=True)
    model_path = 'models/cartpole_dqn.pth'
    torch.save(policy_net.state_dict(), model_path)
    print(f"Model saved to '{model_path}'")

    # Example of loading and evaluating the saved model.
    print("\n--- Evaluation of the saved model ---")
    loaded_policy_net = DQN(n_observations, n_actions).to(device)
    loaded_policy_net.load_state_dict(torch.load(model_path))
    loaded_policy_net.eval()
    
    # You can set render=True here to visualize the agent playing the game.
    evaluate_agent(gym.make("CartPole-v1"), loaded_policy_net, num_episodes=5, render=False)
