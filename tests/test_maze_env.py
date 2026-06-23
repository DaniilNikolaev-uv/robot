"""Мини-тесты согласованности проекта."""

import numpy as np

from maze_env import MazeEnv


def test_reset_returns_gymnasium_tuple():
    env = MazeEnv()
    obs, info = env.reset(seed=123)
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert info["position"] == (0, 0)


def test_wall_collision_penalty_and_position_stays_same():
    env = MazeEnv()
    env.reset(seed=123)
    _, reward, terminated, truncated, info = env.step(2)  # слева граница
    assert reward == -1.0
    assert info["hit_wall"] is True
    assert info["position"] == (0, 0)
    assert terminated is False
    assert truncated is False


def test_known_short_path_reaches_goal():
    env = MazeEnv(max_steps=120)
    env.reset(seed=123)
    # Один корректный путь через лабиринт: вправо x3, вниз x2, вправо x2, вниз x2,
    # вправо x2, вниз x2, влево x2, вниз, вправо x2, вниз, вправо x4.
    actions = [3, 3, 3, 1, 1, 3, 3, 1, 1, 3, 3, 1, 1, 2, 2, 1, 3, 3, 1, 3, 3, 3, 3]
    terminated = False
    total_reward = 0.0
    for action in actions:
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    assert terminated is True
    assert info["success"] is True
    assert info["position"] == (7, 7)
    assert total_reward > 0
