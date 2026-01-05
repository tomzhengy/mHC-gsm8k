"""
REINFORCE training script for mHC gate controller.

Uses simple policy gradient with average-reward baseline (no value network).
This is appropriate for one-step environments (contextual bandits).

Key features:
- Batch collection for variance reduction
- EMA baseline for advantage computation
- Entropy bonus to encourage exploration
- Action distribution logging to detect collapse
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "nanochat-mHC"))

from controller.model_loader import load_frozen_mhc_model
from controller.policy import GatePolicy
from envs.gsm8k_env import GSM8KEnv

# optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def count_actions(action_list: list) -> dict:
    """Count occurrences of each action in a list."""
    counts = defaultdict(int)
    for a in action_list:
        counts[int(a)] += 1
    return dict(counts)


def train_reinforce(
    env: GSM8KEnv,
    policy: GatePolicy,
    optimizer: optim.Optimizer,
    n_episodes: int = 1000,
    batch_size: int = 32,
    beta: float = 0.99,
    entropy_coef: float = 0.01,
    log_interval: int = 10,
    checkpoint_dir: str = None,
    checkpoint_interval: int = 500,
    use_wandb: bool = False,
):
    """
    Train policy using REINFORCE with average-reward baseline.

    Args:
        env: GSM8K environment
        policy: GatePolicy network
        optimizer: PyTorch optimizer
        n_episodes: Total number of episodes to train
        batch_size: Number of episodes per batch
        beta: EMA decay for baseline (0.99 means slow adaptation)
        entropy_coef: Weight for entropy bonus (encourages exploration)
        log_interval: Print logs every N batches
        checkpoint_dir: Directory to save checkpoints (None to skip)
        checkpoint_interval: Save checkpoint every N episodes
        use_wandb: Whether to log to wandb
    """
    device = next(policy.parameters()).device
    baseline = 0.0  # running average reward

    n_batches = n_episodes // batch_size
    total_episodes = 0

    print(f"\nstarting reinforce training")
    print(f"  n_episodes: {n_episodes}")
    print(f"  batch_size: {batch_size}")
    print(f"  n_batches: {n_batches}")
    print(f"  beta: {beta}")
    print(f"  entropy_coef: {entropy_coef}")
    print()

    for batch_idx in range(n_batches):
        # collect batch of episodes
        obs_batch = []
        action_batch = []
        reward_batch = []
        logp_batch = []
        entropy_batch = []

        for _ in range(batch_size):
            obs, info = env.reset()
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)

            # sample action from policy
            action, logp, entropy = policy.get_action(obs_tensor, deterministic=False)

            # take step in environment
            _, reward, _, _, step_info = env.step(action.item())

            obs_batch.append(obs)
            action_batch.append(action)
            reward_batch.append(reward)
            logp_batch.append(logp)
            entropy_batch.append(entropy)

        total_episodes += batch_size

        # compute advantages (reward - baseline)
        rewards = torch.tensor(reward_batch, dtype=torch.float32, device=device)
        advantages = rewards - baseline

        # update baseline with EMA
        batch_mean_reward = rewards.mean().item()
        baseline = beta * baseline + (1 - beta) * batch_mean_reward

        # stack log probs and entropy
        logp = torch.stack(logp_batch)
        entropy = torch.stack(entropy_batch)

        # REINFORCE loss: -log_prob * advantage - entropy_coef * entropy
        policy_loss = -(logp * advantages).mean()
        entropy_loss = -entropy.mean()
        loss = policy_loss + entropy_coef * entropy_loss

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # compute metrics
        accuracy = batch_mean_reward  # reward is 1 for correct, 0 for incorrect
        action_counts = count_actions([a.item() for a in action_batch])

        # logging
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "episode": total_episodes,
                "batch": batch_idx,
                "accuracy": accuracy,
                "baseline": baseline,
                "loss": loss.item(),
                "policy_loss": policy_loss.item(),
                "entropy": entropy.mean().item(),
                "advantage_mean": advantages.mean().item(),
                "advantage_std": advantages.std().item(),
                **{f"action_{k}": v for k, v in action_counts.items()},
            })

        if (batch_idx + 1) % log_interval == 0:
            print(
                f"batch {batch_idx+1}/{n_batches} | "
                f"episodes {total_episodes} | "
                f"accuracy {accuracy:.1%} | "
                f"baseline {baseline:.3f} | "
                f"loss {loss.item():.4f} | "
                f"entropy {entropy.mean().item():.4f} | "
                f"actions {action_counts}"
            )

        # checkpoint
        if checkpoint_dir and (total_episodes % checkpoint_interval == 0):
            checkpoint_path = Path(checkpoint_dir) / f"policy_{total_episodes:06d}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(policy.state_dict(), checkpoint_path)
            print(f"  saved checkpoint to {checkpoint_path}")

    # save final checkpoint
    if checkpoint_dir:
        final_path = Path(checkpoint_dir) / "policy_final.pt"
        torch.save(policy.state_dict(), final_path)
        print(f"saved final checkpoint to {final_path}")

    return policy


def main():
    parser = argparse.ArgumentParser(description="train mhc gate controller with reinforce")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="path to model checkpoint directory")
    parser.add_argument("--step", type=int, default=None,
                        help="checkpoint step to load (default: latest)")
    parser.add_argument("--data_path", type=str, default="data/gsm8k_train.jsonl",
                        help="path to gsm8k training data")
    parser.add_argument("--n_episodes", type=int, default=5000,
                        help="total number of training episodes")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="episodes per batch")
    parser.add_argument("--n_actions", type=int, default=5,
                        choices=[3, 5],
                        help="number of discrete gate actions (3 or 5)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--beta", type=float, default=0.99,
                        help="EMA decay for baseline")
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help="entropy bonus coefficient")
    parser.add_argument("--device", type=str, default="cuda",
                        help="device to use")
    parser.add_argument("--output_dir", type=str, default="controller_checkpoints",
                        help="directory to save policy checkpoints")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="wandb project name (None to disable)")
    parser.add_argument("--wandb_run", type=str, default=None,
                        help="wandb run name")
    args = parser.parse_args()

    # find latest step if not specified
    if args.step is None:
        from nanochat.checkpoint_manager import find_last_step
        args.step = find_last_step(args.checkpoint_dir)
        print(f"using latest checkpoint step: {args.step}")

    # load model
    print(f"loading model from {args.checkpoint_dir} step {args.step}...")
    model, tokenizer, meta = load_frozen_mhc_model(
        args.checkpoint_dir, args.step, args.device
    )
    print(f"model loaded: {meta.get('model_config', {})}")

    # create environment
    env = GSM8KEnv(
        model=model,
        tokenizer=tokenizer,
        data_path=args.data_path,
        device=args.device,
    )

    # adjust action space if n_actions=3
    if args.n_actions == 3:
        env.GATE_VALUES = [0.0, 0.5, 1.0]
        env.action_space = type(env.action_space)(3)
        print(f"using 3 actions: {env.GATE_VALUES}")
    else:
        print(f"using 5 actions: {env.GATE_VALUES}")

    # create policy
    policy = GatePolicy(
        obs_dim=2,
        hidden_dim=64,
        n_actions=args.n_actions,
    ).to(args.device)
    print(f"policy: {policy}")

    # create optimizer
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    # setup wandb
    use_wandb = args.wandb_project is not None and WANDB_AVAILABLE
    if use_wandb:
        run_name = args.wandb_run or f"reinforce_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
        )
        print(f"logging to wandb project: {args.wandb_project}")

    # train
    print("\nstarting training...")
    policy = train_reinforce(
        env=env,
        policy=policy,
        optimizer=optimizer,
        n_episodes=args.n_episodes,
        batch_size=args.batch_size,
        beta=args.beta,
        entropy_coef=args.entropy_coef,
        checkpoint_dir=args.output_dir,
        use_wandb=use_wandb,
    )

    if use_wandb:
        wandb.finish()

    print("\ntraining complete!")


if __name__ == "__main__":
    main()
