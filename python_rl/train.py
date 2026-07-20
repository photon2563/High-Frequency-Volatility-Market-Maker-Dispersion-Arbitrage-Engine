"""
train.py — Training orchestrator for the RL market-making system.

Usage:
    python train.py --agent ppo --episodes 1000
    python train.py --agent sig --episodes 500
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def train_ppo(episodes: int, save_path: str):
    """Train PPO agent on the market-making environment."""
    from env.market_making_env import MarketMakingEnv, MarketConfig
    from agents.ppo_agent import PPOMarketMaker, PPOConfig

    config = MarketConfig(
        initial_price=100.0,
        volatility=0.02,
        base_arrival_rate=10.0,
        max_inventory=50,
        risk_aversion=0.1,
        max_steps=1000,
    )

    env = MarketMakingEnv(config)

    ppo_config = PPOConfig(
        learning_rate=3e-4,
        total_timesteps=episodes * config.max_steps,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01,
    )

    agent = PPOMarketMaker(env, ppo_config)

    print("═" * 60)
    print("Training PPO Market Making Agent")
    print("═" * 60)
    print(f"  Total timesteps: {ppo_config.total_timesteps:,}")
    print(f"  Max steps/episode: {config.max_steps}")
    print(f"  Risk aversion: {config.risk_aversion}")
    print()

    start = time.time()
    agent.train()
    elapsed = time.time() - start

    print(f"\nTraining completed in {elapsed:.1f}s")

    # Evaluate
    eval_results = agent.evaluate(n_episodes=20)
    print("\n─── Evaluation Results ───")
    for k, v in eval_results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save
    agent.save(save_path)
    print(f"\nModel saved to: {save_path}")

    return eval_results


def train_sig(episodes: int, save_path: str):
    """Train Sig-REINFORCE agent on the market-making environment."""
    from env.market_making_env import MarketMakingEnv, MarketConfig
    from agents.sig_reinforce import SigReinforceAgent, SigReinforceConfig

    config = MarketConfig(
        initial_price=100.0,
        volatility=0.02,
        base_arrival_rate=10.0,
        max_inventory=50,
        risk_aversion=0.1,
        max_steps=500,
    )

    env = MarketMakingEnv(config)

    sig_config = SigReinforceConfig(
        sig_depth=3,
        path_length=50,
        n_episodes=episodes,
        learning_rate=1e-3,
        entropy_coef=0.01,
    )

    agent = SigReinforceAgent(env, sig_config)

    print("═" * 60)
    print("Training Sig-REINFORCE Market Making Agent")
    print("═" * 60)
    print(f"  Episodes: {episodes}")
    print(f"  Signature depth: {sig_config.sig_depth}")
    print(f"  Sig dimension: {agent.sig_dim}")
    print()

    start = time.time()
    results = agent.train(n_episodes=episodes)
    elapsed = time.time() - start

    print(f"\nTraining completed in {elapsed:.1f}s")
    print("\n─── Results ───")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train RL market-making agents")
    parser.add_argument(
        "--agent",
        choices=["ppo", "sig"],
        default="ppo",
        help="Agent type: 'ppo' (Stable-Baselines3) or 'sig' (Sig-REINFORCE)",
    )
    parser.add_argument("--episodes", type=int, default=500, help="Training episodes")
    parser.add_argument(
        "--save-path",
        type=str,
        default="models/market_maker",
        help="Path to save trained model",
    )

    args = parser.parse_args()

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)

    if args.agent == "ppo":
        train_ppo(args.episodes, args.save_path)
    elif args.agent == "sig":
        train_sig(args.episodes, args.save_path)


if __name__ == "__main__":
    main()
