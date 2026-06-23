"""Среда лабиринта в стиле Gymnasium для обучения робота.

Клетки лабиринта:
0 - свободная клетка
1 - стена
S - старт агента
G - цель
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MazeEnv(gym.Env):
    """Простая учебная среда лабиринта.

    Агент видит карту стен, своё положение и положение цели.
    Действия: 0 - вверх, 1 - вниз, 2 - влево, 3 - вправо.
    """

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}

    def __init__(self, size: int = 8, max_steps: int = 120, render_mode: Optional[str] = None):
        super().__init__()
        self.size = size
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Фиксированная карта делает проект воспроизводимым и понятным.
        self.maze = np.array(
            [
                [0, 0, 0, 0, 1, 0, 0, 0],
                [1, 1, 1, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 1, 1, 1, 1, 0, 1, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 1, 0, 1, 1, 1, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1, 1, 1, 0],
            ],
            dtype=np.float32,
        )
        if self.maze.shape != (size, size):
            raise ValueError("В учебном проекте готовая карта имеет размер 8x8. Используйте size=8.")

        self.start_pos = np.array([0, 0], dtype=np.int64)
        self.goal_pos = np.array([7, 7], dtype=np.int64)
        self.agent_pos = self.start_pos.copy()
        self.steps = 0

        # Наблюдение: 3 канала size x size: стены, агент, цель.
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3, size, size), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

    def _get_obs(self) -> np.ndarray:
        """Собираем состояние, которое получает нейросеть."""
        obs = np.zeros((3, self.size, self.size), dtype=np.float32)
        obs[0] = self.maze
        obs[1, self.agent_pos[0], self.agent_pos[1]] = 1.0
        obs[2, self.goal_pos[0], self.goal_pos[1]] = 1.0
        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Начинаем новый эпизод: агент снова стоит на старте."""
        super().reset(seed=seed)
        self.agent_pos = self.start_pos.copy()
        self.steps = 0
        info = {"position": tuple(map(int, self.agent_pos)), "goal": tuple(map(int, self.goal_pos))}
        return self._get_obs(), info

    def step(self, action: int):
        """Делаем один шаг агента и возвращаем результат.

        Возвращается формат Gymnasium:
        observation, reward, terminated, truncated, info
        """
        self.steps += 1
        old_pos = self.agent_pos.copy()

        moves = {
            0: np.array([-1, 0]),  # вверх
            1: np.array([1, 0]),   # вниз
            2: np.array([0, -1]),  # влево
            3: np.array([0, 1]),   # вправо
        }
        if int(action) not in moves:
            raise ValueError("Действие должно быть числом от 0 до 3.")

        new_pos = self.agent_pos + moves[int(action)]
        hit_wall = False

        # Если агент вышел за границу или упёрся в стену, он остаётся на месте.
        outside = np.any(new_pos < 0) or np.any(new_pos >= self.size)
        if outside or self.maze[new_pos[0], new_pos[1]] == 1:
            new_pos = old_pos
            hit_wall = True

        old_distance = np.abs(old_pos - self.goal_pos).sum()
        self.agent_pos = new_pos
        new_distance = np.abs(self.agent_pos - self.goal_pos).sum()

        terminated = bool(np.array_equal(self.agent_pos, self.goal_pos))
        truncated = self.steps >= self.max_steps and not terminated

        # Система наград: маленький штраф за каждый шаг, большой штраф за стену,
        # небольшая подсказка за приближение к цели и большая награда за достижение цели.
        reward = -0.05 + 0.10 * float(old_distance - new_distance)
        if hit_wall:
            reward = -1.0
        if terminated:
            reward = 10.0

        info = {
            "position": tuple(map(int, self.agent_pos)),
            "hit_wall": hit_wall,
            "steps": self.steps,
            "success": terminated,
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        """Показываем лабиринт в консоли."""
        symbols = []
        for r in range(self.size):
            row = []
            for c in range(self.size):
                pos = np.array([r, c])
                if np.array_equal(pos, self.agent_pos):
                    row.append("A")
                elif np.array_equal(pos, self.goal_pos):
                    row.append("G")
                elif self.maze[r, c] == 1:
                    row.append("#")
                else:
                    row.append(".")
            symbols.append(" ".join(row))
        text = "\n".join(symbols)
        if self.render_mode == "human":
            print(text)
            print()
        return text
