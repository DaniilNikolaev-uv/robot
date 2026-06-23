"""Обучение DQN-агента в лабиринте.

Запуск:
python train.py --episodes 500
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dqn_agent import DQNAgent, DQNConfig
from maze_env import MazeEnv
from utils import ensure_dirs, save_training_plots, set_seed


def evaluate_agent(agent: DQNAgent, max_steps: int = 120, seed: int = 123) -> tuple[bool, int, float]:
    """Проверяем агента без случайных действий, как в настоящем инференсе."""
    eval_env = MazeEnv(max_steps=max_steps)
    state, _ = eval_env.reset(seed=seed)
    total_reward = 0.0
    for step in range(1, max_steps + 1):
        action = agent.select_action(state, training=False)
        state, reward, terminated, truncated, _ = eval_env.step(action)
        total_reward += reward
        if terminated or truncated:
            return bool(terminated), step, total_reward
    return False, max_steps, total_reward


def train(episodes: int = 500, seed: int = 42, max_steps: int = 120, model_path: str = "models/dqn_maze.pth"):
    """Основной цикл обучения агента."""
    set_seed(seed)
    ensure_dirs()

    env = MazeEnv(max_steps=max_steps)
    obs, _ = env.reset(seed=seed)

    config = DQNConfig(
        gamma=0.99,
        learning_rate=1e-3,
        batch_size=64,
        replay_capacity=10_000,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.992,
        target_update_every=20,
    )
    agent = DQNAgent(env.observation_space.shape, env.action_space.n, config=config)

    rewards_history: list[float] = []
    lengths_history: list[int] = []
    best_eval_steps = max_steps + 1
    best_eval_reward = -float("inf")

    for episode in range(1, episodes + 1):
        state, _ = env.reset(seed=seed + episode)
        episode_reward = 0.0
        last_loss = None

        for step in range(1, max_steps + 1):
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.remember(state, action, reward, next_state, done)
            loss = agent.optimize_model()
            if loss is not None:
                last_loss = loss

            state = next_state
            episode_reward += reward

            if done:
                break

        agent.decay_epsilon()
        if episode % config.target_update_every == 0:
            agent.update_target_network()

        rewards_history.append(episode_reward)
        lengths_history.append(step)

        # Сохраняем модель только после отдельной greedy-проверки.
        # Это важно: во время обучения агент иногда доходит до цели случайно из-за epsilon-greedy.
        if episode % 10 == 0:
            eval_success, eval_steps, eval_reward = evaluate_agent(agent, max_steps=max_steps, seed=seed)
            if eval_success and (eval_steps < best_eval_steps or eval_reward > best_eval_reward):
                best_eval_steps = eval_steps
                best_eval_reward = eval_reward
                agent.save(model_path)
        else:
            eval_success, eval_steps, eval_reward = False, 0, 0.0

        if episode == 1 or episode % 10 == 0:
            status = "успех" if info.get("success") else "не дошёл"
            loss_text = "нет" if last_loss is None else f"{last_loss:.4f}"
            print(
                f"Эпизод {episode:4d}/{episodes} | "
                f"награда {episode_reward:7.2f} | шаги {step:3d} | "
                f"epsilon {agent.epsilon:.3f} | loss {loss_text} | {status} | "
                f"eval: {'успех' if eval_success else 'нет'} {eval_steps} шаг."
            )

    save_training_plots(rewards_history, lengths_history)

    print("\nОбучение завершено.")
    print(f"Модель сохранена: {Path(model_path).resolve()}")
    print(f"График наград: {Path('results/rewards.png').resolve()}")
    print(f"График длины эпизодов: {Path('results/episode_lengths.png').resolve()}")
    return rewards_history, lengths_history


def parse_args():
    parser = argparse.ArgumentParser(description="Обучение DQN-агента в учебном лабиринте")
    parser.add_argument("--episodes", type=int, default=500, help="Количество эпизодов обучения, рекомендуется 300-1000")
    parser.add_argument("--seed", type=int, default=42, help="Random seed для воспроизводимости")
    parser.add_argument("--max-steps", type=int, default=120, help="Лимит шагов в эпизоде")
    parser.add_argument("--model-path", type=str, default="models/dqn_maze.pth", help="Куда сохранить веса модели")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(episodes=args.episodes, seed=args.seed, max_steps=args.max_steps, model_path=args.model_path)
