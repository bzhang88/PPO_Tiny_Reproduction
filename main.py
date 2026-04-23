import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List

gamma = 0.99
lr = 1e-3
eps_clip = 0.2
episodes = 1000
hidden_size = 64
num_epochs = 4
seed = 42

torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)


def compute_returns(rewards):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


def train(method, env_name):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = Policy(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    reward_history = []

    print(f"Training {method} on {env_name}...")

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        states = []
        actions = []
        old_output = []
        rewards = []
        done = False

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            output = policy(state_tensor)
            dist = Categorical(output)
            action = dist.sample()

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            states.append(state_tensor)
            actions.append(action)
            old_output.append(dist.log_prob(action))
            rewards.append(reward)

            state = next_state

        value = compute_returns(rewards)

        if method == "PPO":
            old_output = torch.stack(old_output).detach()
            states_t = torch.stack(states)
            actions_t = torch.stack(actions)
            for _ in range(num_epochs):
                output = policy(states_t)

                dist = Categorical(output)
                entropy = dist.entropy().mean()

                new_output = dist.log_prob(actions_t)
                ratio = torch.exp(new_output - old_output)
                clipped_ratio = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip)

                loss = (
                    -torch.min(ratio * value, clipped_ratio * value).mean()
                    - 0.01 * entropy
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        elif method == "REINFORCE":
            log_probs = torch.stack(old_output)
            loss = -(log_probs * value).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_reward = sum(rewards)
        reward_history.append(total_reward)

        if episode % 200 == 0 or episode == episodes - 1:
            print(f"[{method}] Episode {episode}, Reward: {total_reward:.1f}")

    env.close()
    return reward_history


if os.path.exists("results"):
    print("Directory 'results' already exists. Plots will be saved there.")
else:
    os.makedirs("results", exist_ok=True)

envs = ["CartPole-v1", "Acrobot-v1"]

for env_name in envs:
    ppo_rewards = train("PPO", env_name)
    reinforce_rewards = train("REINFORCE", env_name)

    plt.figure(figsize=(10, 6))
    plt.plot(ppo_rewards, label="PPO", color="tab:blue")
    plt.plot(reinforce_rewards, label="REINFORCE", color="tab:orange")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"PPO vs REINFORCE on {env_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(
        f"results/comparison_{env_name.lower().replace('-', '')}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved plot: results/comparison_{env_name.lower().replace('-', '')}.png\n\n")

print("All training completed. Plots saved in the 'results' directory.")
