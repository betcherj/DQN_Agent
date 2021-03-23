import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import gym
import time
import os
import numpy as np
import argparse
from tqdm import tqdm


seed = 0
device = torch.device("cpu")
torch.manual_seed(seed)


class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size):
        self.action_size = action_size
        self.memory = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.experience = 0


    def __len__(self):
        return len(self.memory)


    def add(self, state, action, reward, next_state, done):
        if len(self.memory)>self.buffer_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))


    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e[0] is not None])).float()
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e[1] is not None])).long()
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e[2] is not None])).float()
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e[3] is not None])).float()
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e[4] is not None]).astype(np.uint8)).float()
        return (states, actions, rewards, next_states, dones)


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_layer_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size*2) #Not sure reason for this *2
        self.fc3 = nn.Linear(hidden_layer_size*2, action_size)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Agent:
    def __init__(self, state_size, action_size, replay_memory_size, batch_size, gamma,
                 alpha, update_rate, tau):
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = replay_memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.update_rate = update_rate
        self.network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.alpha)
        self.criterion = torch.nn.MSELoss()
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size)
        self.t_step = 0


    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step += 1
        if self.t_step % self.update_rate == 0 and len(self.memory) > self.batch_size:
            self.t_step = 0
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma)


    def getAction(self, state, epsilon):
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.network.eval() #Set local network to eval, we are not learning here
        with torch.no_grad():
            action_values = self.network.forward(state)
            best_action = np.argmax(action_values.data.numpy())
        self.network.train()

        if random.random() <= epsilon:
            return random.randint(0, self.action_size-1)
        else:
            return best_action


    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        Qsa = self.network(states).gather(1, actions)

        self.network.eval()
        with torch.no_grad():
            # Qsa_prime_target = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
            actions_q_local = self.network(next_states).detach().max(1)[1].unsqueeze(1).long()
            Qsa_prime_targets = self.target_network(next_states).gather(1, actions_q_local)
        self.network.train()

        #Qsa_targets = rewards + (gamma * Qsa_prime_targets * (1 - dones))
        Qsa_targets = rewards + (gamma * Qsa_prime_targets)
        loss = self.criterion(Qsa, Qsa_targets)

        self.optimizer.zero_grad() #Zeros out gradient, default behavior is to accumulate
        loss.backward()
        self.optimizer.step()

        for target_param, local_param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


class Solver:
    def __init__(self, replay_memory_size=10000, batch_size=64, gamma=0.99,
                 alpha=.001, update_rate=4, tau=.1):
        self.num_episodes = 4000
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.training_rewards = []
        self.testing_rewards = []
        self.replay_memory_size = replay_memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self. alpha = alpha
        self.update_rate = update_rate
        self.tau = tau
        self.env = gym.make('LunarLander-v2')
        self.agent = Agent(len(self.env.observation_space.sample()), self.env.action_space.n, self.replay_memory_size,
                           self.batch_size, self.gamma, self.alpha, self.update_rate, self.tau)
        self.training_time = 0.0
        self.average_steps_per_episode_training = None
        self.hit_rolling_average_target = False
        self.testing_time = 0.0
        self.average_steps_per_episode_testing = None



    def train(self):
        print("Training DQN Agent...")
        pbar = tqdm(total=self.num_episodes, leave=False)
        epsilon = self.epsilon
        start = time.time()
        for episode in range(self.num_episodes):
            observation = self.env.reset()
            cumulative_reward = 0
            done = False
            episode_steps = 0
            while not done:
                action = self.agent.getAction(observation, epsilon)
                next_observation, reward, done, info = self.env.step(action)
                episode_steps+=1
                if episode_steps == self.env._max_episode_steps:
                    self.agent.step(observation, action, reward, next_observation, 0) #Deal with the case of artifical terminiation
                else:
                    self.agent.step(observation, action, reward, next_observation, done)
                observation = next_observation
                cumulative_reward += reward

            if self.average_steps_per_episode_training:
                self.average_steps_per_episode_training += episode_steps
                self.average_steps_per_episode_training /= 2.0
            else:
                self.average_steps_per_episode_training = episode_steps
            self.training_rewards.append(cumulative_reward)
            if not self.hit_rolling_average_target and episode_steps>99:
                self.hit_rolling_average_target = np.mean(self.training_rewards[-100:]) >= 200
            epsilon = max(self.epsilon_min, self.epsilon_decay*epsilon)
            pbar.update(1)
        end = time.time()
        self.training_time = end-start
        pbar.close()
        print()


    def test(self):
        print("Evaluating DQN Agent...")
        pbar = tqdm(total=100, leave=False)
        start = time.time()
        for episode in range(100):
            observation = self.env.reset()
            cumulative_reward = 0
            episode_steps = 0
            done = False
            while not done:
                action = self.agent.getAction(observation, 0)
                next_observation, reward, done, info = self.env.step(action)
                observation = next_observation
                cumulative_reward += reward
                episode_steps += 1
            if self.average_steps_per_episode_testing:
                self.average_steps_per_episode_testing += episode_steps
                self.average_steps_per_episode_testing /= 2.0
            else:
                self.average_steps_per_episode_testing = episode_steps
            self.testing_rewards.append(cumulative_reward)
            pbar.update(1)
        end = time.time()
        self.testing_time = end-start
        pbar.close()
        print()


    def summarize(self, save_to_disc=False):
        if not self.testing_rewards or not self.testing_rewards:
            print("Untrained or untested agent!!")
            return

        print("################Hyper Params################")
        print("--------------------Agent------------------")
        print("Num Episodes: " + str(self.num_episodes))
        print("Epsilon: " + str(self.epsilon))
        print("Epsilon Decay: " + str(self.epsilon_decay))
        print("Epsilon Min: " + str(self.epsilon_min))
        print("-----------------Replay Buffer--------------")
        print("Replay Memory Size: " + str(self.replay_memory_size))
        print("Batch Size: " + str(self.batch_size))
        print("-------------------QNetworks----------------")
        print("Alpha: " + str(self.alpha))
        print("Gamma: " + str(self.gamma))
        print("Update Rate: " + str(self.update_rate))
        print("Tau: " + str(self.tau))
        print()
        print("##################Training##################")
        print("Time: " + str(self.training_time))
        print("Average Reward: " + str(np.mean(self.training_rewards)))
        print("Average Steps per Episode: " + str(self.average_steps_per_episode_training))
        print("Hit Rolling Average Target: " + str(self.hit_rolling_average_target))
        print()
        print("##################Testing###################")
        print("Time: " + str(self.testing_time))
        print("Average Reward: " + str(np.mean(self.testing_rewards)))
        print("Average Steps per Episode: " + str(self.average_steps_per_episode_testing))
        print()

        if save_to_disc:
            torch.save(self.agent, "./trained_agent.pt")

        train_episodes = np.arange(len(self.training_rewards))
        plt.plot(train_episodes, self.training_rewards)
        plt.ylabel('Reward', fontsize=15)
        plt.xlabel('Episode', fontsize=15)
        plt.title("Training Data")
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.savefig("./training_rewards.png")
        plt.show()

        test_episodes = np.arange(len(self.testing_rewards))
        plt.plot(test_episodes, self.testing_rewards)
        plt.ylabel('Reward', fontsize=15)
        plt.xlabel('Episode', fontsize=15)
        plt.title("Testing Data")
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.savefig("./testing_rewards.png")
        plt.show()


if __name__ == "__main__":
    sol = Solver()
    sol.train()
    sol.test()
    sol.summarize()
