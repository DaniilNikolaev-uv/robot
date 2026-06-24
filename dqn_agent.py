"""DQN-агент для обучения навигации в лабиринте."""

from __future__ import annotations

import random
from collections import deque, namedtuple
from dataclasses import dataclass
from typing import Deque

import numpy as np
import torch
from torch import nn, optim


Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class ReplayBuffer:
    """Буфер опыта: хранит прошлые шаги агента для обучения DQN."""

    def __init__(self, capacity: int):
        self.memory: Deque[Transition] = deque(maxlen=capacity)

    def push(self, state, action: int, reward: float, next_state, done: bool) -> None:
        self.memory.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class DQNNetwork(nn.Module):
    """Нейросеть, которая оценивает полезность каждого действия.

    Вход: 3 канала карты 8x8: стены, агент, цель.
    Выход: 4 Q-значения для действий вверх/вниз/влево/вправо.
    """

    def __init__(self, input_shape: tuple[int, int, int], n_actions: int):
        super().__init__()
        channels, height, width = input_shape
        flat_size = channels * height * width
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class DQNConfig:
    gamma: float = 0.99
    learning_rate: float = 1e-3
    batch_size: int = 64
    replay_capacity: int = 10_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    target_update_every: int = 20


class DQNAgent:
    """Агент DQN с replay buffer, epsilon-greedy и target network."""

    def __init__(self, observation_shape, n_actions: int, config: DQNConfig | None = None, device: str | None = None):
        self.config = config or DQNConfig()
        self.n_actions = n_actions
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.policy_net = DQNNetwork(observation_shape, n_actions).to(self.device)
        self.target_net = DQNNetwork(observation_shape, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.learning_rate)
        self.memory = ReplayBuffer(self.config.replay_capacity)
        self.epsilon = self.config.epsilon_start
        self.loss_fn = nn.SmoothL1Loss()

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Выбираем действие.

        Во время обучения иногда выбирается случайное действие, чтобы агент исследовал лабиринт.
        Во время проверки берём лучшее действие по мнению нейросети.
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def remember(self, state, action: int, reward: float, next_state, done: bool) -> None:
        self.memory.push(state, action, reward, next_state, done)

    def optimize_model(self) -> float | None:
        """Обновляем веса основной сети на случайной пачке опыта."""
        if len(self.memory) < self.config.batch_size:
            return None

        transitions = self.memory.sample(self.config.batch_size)
        batch = Transition(*zip(*transitions))

        states = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s, a) для действий, которые реально сделал агент.
        current_q = self.policy_net(states).gather(1, actions)

        # Target network даёт более стабильную оценку будущей награды.
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (1.0 - dones) * self.config.gamma * next_q

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        return float(loss.item())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

    def update_target_network(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str) -> None:
        torch.save(
            {
                "model_state_dict": self.policy_net.state_dict(),
                "epsilon": self.epsilon,
                "config": self.config.__dict__,
            },
            path,
        )

    def load(self, path: str, map_location: str | None = None) -> None:
        checkpoint = torch.load(
            path,
            map_location=map_location or self.device,
            weights_only=True,
        )
        self.policy_net.load_state_dict(checkpoint["model_state_dict"])
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.epsilon = float(checkpoint.get("epsilon", self.config.epsilon_end))
