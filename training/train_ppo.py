#!/usr/bin/env python3
"""
Full GPU PPO Training for Quadrotor Control

Usage:
    python train_ppo.py                    # Default training
    python train_ppo.py --timesteps 10M   # 10 million timesteps
    python train_ppo.py --resume path/to/checkpoint.pt  # Resume training
    python train_ppo.py --export           # Export after training
"""

import argparse
import os
import sys
import torch
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ppo_gpu import GPUPPO, PPOConfig
from ppo_gpu.export import export_to_c_header, export_onnx, validate_export


def parse_args():
    parser = argparse.ArgumentParser(description="GPU PPO Training for Quadrotor")
    
    # Training
    parser.add_argument(
        "--timesteps", type=str, default="50M",
        help="Total timesteps (e.g., 10M, 50M, 100M)"
    )
    parser.add_argument(
        "--num-envs", type=int, default=512,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--rollout-steps", type=int, default=512,
        help="Steps per rollout"
    )
    
    # PPO hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-epsilon", type=float, default=0.2, help="PPO clip")
    parser.add_argument("--ppo-epochs", type=int, default=10, help="PPO epochs")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden layer size")
    
    # Checkpoints
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Directory for checkpoints (default: auto-generated)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    
    # Export
    parser.add_argument(
        "--export", action="store_true",
        help="Export policy after training"
    )
    parser.add_argument(
        "--export-path", type=str, default="controller/actor.h",
        help="Path for exported C header"
    )
    
    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval", action="store_true", help="Run evaluation only")
    
    return parser.parse_args()


def parse_timesteps(s: str) -> int:
    """Parse timestep string like '10M' or '1000000'"""
    s = s.upper().strip()
    if s.endswith('M'):
        return int(float(s[:-1]) * 1_000_000)
    elif s.endswith('K'):
        return int(float(s[:-1]) * 1_000)
    else:
        return int(s)


def main():
    args = parse_args()
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires a GPU.")
        sys.exit(1)
    
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    # Create checkpoint directory
    if args.checkpoint_dir is None:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        args.checkpoint_dir = f"checkpoints/ppo_gpu_{timestamp}"
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(f"Checkpoints: {args.checkpoint_dir}")
    
    # Create config
    config = PPOConfig(
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        total_timesteps=parse_timesteps(args.timesteps),
        learning_rate=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        ppo_epochs=args.ppo_epochs,
        hidden_dim=args.hidden_dim,
        seed=args.seed
    )
    
    # Print config
    print("Configuration:")
    print(f"  Total timesteps: {config.total_timesteps:,}")
    print(f"  Num envs: {config.num_envs}")
    print(f"  Rollout steps: {config.rollout_steps}")
    print(f"  Batch size: {config.num_envs * config.rollout_steps:,}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print()
    
    # Create PPO trainer
    ppo = GPUPPO(
        config=config,
        device=device,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Resume from checkpoint
    if args.resume:
        ppo.load_checkpoint(args.resume)
    
    # Evaluation only
    if args.eval:
        print("Running evaluation...")
        eval_stats = ppo.evaluate(num_episodes=100)
        print(f"Evaluation Results:")
        print(f"  Mean Return: {eval_stats['mean_return']:.2f}")
        print(f"  Std Return: {eval_stats['std_return']:.2f}")
        print(f"  Min Return: {eval_stats['min_return']:.2f}")
        print(f"  Max Return: {eval_stats['max_return']:.2f}")
        return
    
    # Train
    print("="*60)
    print("Starting Training")
    print("="*60)
    
    history = ppo.train()
    
    # Export
    if args.export:
        print()
        print("="*60)
        print("Exporting Policy")
        print("="*60)
        
        # Get observation statistics
        obs_mean = ppo.env.obs_mean
        obs_std = torch.sqrt(ppo.env.obs_var + 1e-8)
        
        # Export to C header
        c_path = export_to_c_header(
            ppo.policy,
            obs_mean,
            obs_std,
            args.export_path
        )
        
        # Export to ONNX
        onnx_path = args.export_path.replace('.h', '.onnx')
        export_onnx(ppo.policy, obs_mean, obs_std, onnx_path)
        
        # Validate export
        print("\nValidating export...")
        validate_export(ppo.policy, obs_mean, obs_std, c_path)
    
    print()
    print("Training complete!")


if __name__ == "__main__":
    main()
