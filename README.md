# Autonomous Agent for CartPole-v1 using Deep Q-Networks (DQN)

This project demonstrates the implementation of a **Deep Q-Network (DQN)** agent to solve the classic **CartPole-v1** environment from Gymnasium. The agent learns to balance a pole on a cart by applying forces to move the cart left or right.

##  Key Features
- **Reinforcement Learning:** Implements a Deep Q-Network (DQN) to learn an optimal policy.
- **Experience Replay:** Utilizes a replay buffer to stabilize training and improve sample efficiency.
- **Target Network:** Employs a separate target network to provide a stable learning target.
- **Epsilon-Greedy Policy:** Balances exploration and exploitation to discover the environment's dynamics.
- **PyTorch Implementation:** Built entirely using the PyTorch deep learning framework.

##  How to Run the Project

### Prerequisites
- Python 3.8+
- The following libraries (install with `pip install -r requirements.txt`):
  - gymnasium
  - torch
  - matplotlib
  - numpy

### Training the Agent
To train the agent and save the model weights, simply run the main script:
```bash
python main.py
