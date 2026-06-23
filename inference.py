"""Проверка обученной модели DQN в лабиринте.

Запуск:
python inference.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from dqn_agent import DQNAgent
from maze_env import MazeEnv
from utils import set_seed


def run_inference(model_path: str = "models/dqn_maze.pth", seed: int = 42, delay: float = 0.25, max_steps: int = 120):
    """Загружаем модель и показываем путь агента в консоли."""
    set_seed(seed)

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Файл модели не найден: {model_path}. Сначала запустите обучение: python train.py --episodes 500"
        )

    env = MazeEnv(max_steps=max_steps, render_mode="human")
    agent = DQNAgent(env.observation_space.shape, env.action_space.n)
    agent.load(model_path, map_location=str(agent.device))
    agent.epsilon = 0.0  # Во время инференса агент не выбирает случайные действия.

    state, _ = env.reset(seed=seed)
    total_reward = 0.0
    success = False

    print("Начальное состояние лабиринта:")
    env.render()

    action_names = {0: "вверх", 1: "вниз", 2: "влево", 3: "вправо"}

    for step in range(1, max_steps + 1):
        action = agent.select_action(state, training=False)
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(f"Шаг {step}: действие = {action_names[action]}, награда = {reward:.2f}")
        env.render()
        if delay > 0:
            time.sleep(delay)

        if terminated or truncated:
            success = bool(terminated)
            break

    print("Итог проверки:")
    if success:
        print("Агент дошёл до цели.")
    else:
        print("Агент не дошёл до цели за лимит шагов.")
    print(f"Количество шагов: {step}")
    print(f"Итоговая награда: {total_reward:.2f}")
    return success, step, total_reward


def parse_args():
    parser = argparse.ArgumentParser(description="Проверка обученного DQN-агента")
    parser.add_argument("--model-path", type=str, default="models/dqn_maze.pth", help="Путь к сохранённой модели")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--delay", type=float, default=0.25, help="Пауза между шагами в секундах")
    parser.add_argument("--max-steps", type=int, default=120, help="Лимит шагов")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(model_path=args.model_path, seed=args.seed, delay=args.delay, max_steps=args.max_steps)
