"""
ppo_agent.py — PPO agent for options market making.

Uses Stable-Baselines3 PPO with a custom policy network.
Entropy regularization ensures exploration of diverse bid-ask strategies.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class PPOConfig:
    """Hyperparameters for PPO training."""
    learning_rate: float = 3e-4
    n_steps: int = 2048            # Steps per rollout
    batch_size: int = 64           # Minibatch size
    n_epochs: int = 10             # PPO epochs per update
    gamma: float = 0.99            # Discount factor
    gae_lambda: float = 0.95       # GAE lambda
    clip_range: float = 0.2        # PPO clip range
    ent_coef: float = 0.01         # Entropy coefficient (exploration)
    vf_coef: float = 0.5           # Value function coefficient
    max_grad_norm: float = 0.5     # Gradient clipping
    total_timesteps: int = 500_000 # Total training steps

    # Network architecture
    net_arch: list = None          # Policy/value network hidden layers

    def __post_init__(self):
        if self.net_arch is None:
            self.net_arch = [dict(pi=[128, 128], vf=[128, 128])]


class PPOMarketMaker:
    """
    PPO-based market making agent.

    Wraps Stable-Baselines3's PPO implementation with custom configuration
    for the market-making domain.
    """

    def __init__(self, env, config: Optional[PPOConfig] = None):
        self.env = env
        self.config = config or PPOConfig()
        self.model = None

    def build(self):
        """Initialize the PPO model."""
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.callbacks import EvalCallback
        except ImportError:
            raise ImportError(
                "stable-baselines3 is required. Install with: "
                "pip install stable-baselines3"
            )

        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            ent_coef=self.config.ent_coef,
            vf_coef=self.config.vf_coef,
            max_grad_norm=self.config.max_grad_norm,
            policy_kwargs=dict(net_arch=self.config.net_arch),
            verbose=1,
        )
        return self

    def train(self, total_timesteps: Optional[int] = None, callback=None):
        """Train the PPO agent."""
        if self.model is None:
            self.build()

        steps = total_timesteps or self.config.total_timesteps
        self.model.learn(total_timesteps=steps, callback=callback)
        return self

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        """Predict action given observation."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

    def save(self, path: str):
        """Save model to disk."""
        if self.model is not None:
            self.model.save(path)

    def load(self, path: str):
        """Load model from disk."""
        from stable_baselines3 import PPO
        self.model = PPO.load(path, env=self.env)
        return self

    def evaluate(self, n_episodes: int = 10) -> Dict[str, Any]:
        """Evaluate the agent over multiple episodes."""
        if self.model is None:
            raise RuntimeError("Model not trained.")

        episode_rewards = []
        episode_lengths = []
        final_inventories = []

        for _ in range(n_episodes):
            obs, info = self.env.reset()
            total_reward = 0.0
            steps = 0
            done = False

            while not done:
                action = self.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            final_inventories.append(info.get("inventory", 0))

        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "mean_abs_final_inventory": np.mean(np.abs(final_inventories)),
            "n_episodes": n_episodes,
        }
