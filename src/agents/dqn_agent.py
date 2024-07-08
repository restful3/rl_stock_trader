import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import torch.multiprocessing as mp
from config import config

class DQN(nn.Module):
    def __init__(self, state_size, action_size, net_type='dnn', lstm_sequence_length=5):
        super(DQN, self).__init__()
        self.net_type = net_type
        self.lstm_sequence_length = lstm_sequence_length
        if net_type == 'dnn':
            self.fc1 = nn.Linear(state_size, 24)
            self.fc2 = nn.Linear(24, 24)
            self.fc3 = nn.Linear(24, action_size)
        elif net_type == 'lstm':
            self.lstm = nn.LSTM(state_size, 24, batch_first=True)
            self.fc = nn.Linear(24, action_size)

    def forward(self, x):
        if self.net_type == 'dnn':
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)
        elif self.net_type == 'lstm':
            batch_size = x.size(0)
            x = x.view(batch_size, self.lstm_sequence_length, -1)
            lstm_out, _ = self.lstm(x)
            return self.fc(lstm_out[:, -1, :])


class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, memory_size=2000,
                 target_update_frequency=100, net_type='dnn', lstm_sequence_length=5):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.model = DQN(state_size, action_size, net_type)
        self.target_model = DQN(state_size, action_size, net_type)
        self.update_target_model()
        self.model.share_memory()
        self.target_model.share_memory()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.num_processes = config['num_processes']
        self.batch_size_per_process = config['batch_size_per_process']
        self.target_update_frequency = target_update_frequency
        self.update_counter = 0
        self.net_type = net_type
        self.lstm_sequence_length = lstm_sequence_length
        self.model = DQN(state_size, action_size, net_type, lstm_sequence_length)
        self.target_model = DQN(state_size, action_size, net_type, lstm_sequence_length)


    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        if self.net_type == 'lstm':
            state = state.view(1, self.lstm_sequence_length, -1)
        with torch.no_grad():
            q_values = self.model(state).squeeze()
        return torch.argmax(q_values).item()

    def _replay_process(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1)

        if self.net_type == 'lstm':
            states = states.view(-1, self.lstm_sequence_length, states.size(-1))
            next_states = next_states.view(-1, self.lstm_sequence_length, next_states.size(-1))
        
        current_q_values = self.model(states).gather(1, actions)
        next_q_values = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)
        expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(current_q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, min(batch_size, len(self.memory)))
        loss = self._replay_process(batch)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.update_counter += 1
        if self.update_counter % self.target_update_frequency == 0:
            self.update_target_model()

        return loss

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'net_type': self.net_type
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.net_type = checkpoint['net_type']
        self.update_target_model()
        self.model.eval()

if __name__ == "__main__":
    # 여기에 학습 및 평가 로직을 추가할 수 있습니다.
    pass