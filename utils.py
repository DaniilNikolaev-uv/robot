"""Вспомогательные функции для проекта DQN-лабиринта."""

from __future__ import annotations

import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Фиксируем случайность, чтобы результаты было легче повторить."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    """Создаём папки для весов и результатов, если их ещё нет."""
    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)


def moving_average(values: list[float], window: int = 20) -> np.ndarray:
    """Скользящее среднее делает график наград более гладким."""
    if len(values) == 0:
        return np.array([])
    window = max(1, min(window, len(values)))
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def save_training_plots(rewards: list[float], lengths: list[int], output_dir: str = "results") -> None:
    """Сохраняем два графика: награды и длины эпизодов."""
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.35, label="Награда за эпизод")
    if len(rewards) >= 5:
        avg = moving_average(rewards, window=20)
        plt.plot(range(len(rewards) - len(avg), len(rewards)), avg, label="Скользящее среднее")
    plt.title("DQN: награда по эпизодам")
    plt.xlabel("Эпизод")
    plt.ylabel("Суммарная награда")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "rewards.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(lengths, label="Количество шагов")
    plt.title("DQN: длина прохождения по эпизодам")
    plt.xlabel("Эпизод")
    plt.ylabel("Шаги")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "episode_lengths.png", dpi=150)
    plt.close()
