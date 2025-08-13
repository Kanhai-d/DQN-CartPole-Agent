# Autonomous Agent for CartPole-v1 using Deep Q-Networks (DQN)

This project showcases a reinforcement learning agent built to master the classic **CartPole-v1** challenge from the Gymnasium library.  
In this environment, a pole is attached to a moving cart, and the agent must learn to apply forces to the cart to keep the pole balanced for as long as possible.

## Project Overview
The agent is trained using a **Deep Q-Network (DQN)** — a deep learning approach that estimates action values (Q-values) and learns the optimal policy for balancing the pole.  
Training involves repeatedly playing the game, collecting experiences, and improving the decision-making policy over time.

### Key Concepts Used
- **Deep Q-Network (DQN)**: A neural network that approximates the Q-value function for each possible action.
- **Experience Replay**: Stores past experiences in a replay buffer and samples them randomly to break correlations in training data.
- **Target Network**: A separate network to stabilize learning updates.
- **Epsilon-Greedy Policy**: Balances exploration of new strategies and exploitation of known strategies.
- **PyTorch Framework**: Used to design, train, and evaluate the neural network.

## Learning Process
1. The environment provides the agent with its current state (cart position, velocity, pole angle, and angular velocity).
2. The agent selects an action (push left or push right) based on the policy.
3. The environment returns the next state and a reward.
4. The agent stores this experience and uses it to improve its policy over many episodes.

## Outcome
After sufficient training, the agent learns a policy that keeps the pole balanced for the maximum allowed duration in the environment.  
The project demonstrates the core principles of reinforcement learning in a simple, visual, and engaging task.
