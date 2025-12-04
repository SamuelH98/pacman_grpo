"""
GRPO Training for Pacman - FIXED VERSION

"""
import argparse
import sys
import os
import shutil
from random import random, randint, sample
from multiprocessing import Pool, cpu_count
from functools import partial
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tensorboardX import SummaryWriter
from tqdm import tqdm

from src.grpo_network import GRPOPolicyNetwork
from src.Pacman_Complete.run import GameController as Pacman
from src.Pacman_Complete.pellets import PelletGroup
from src.utils import pre_processing


def get_args():
    parser = argparse.ArgumentParser(
        """GRPO to play Pacman""")
    parser.add_argument("--image_size", type=int, default=84, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=50, help="The number of samples per mini-batch")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epoch", type=int, default=3, help="Number of policy update epochs")
    parser.add_argument("--T_horizon", type=int, default=1024, help="Maximum trajectory length")
    parser.add_argument("--num_rollouts", type=int, default=8, help="Number of rollouts per group (G)")
    parser.add_argument("--kl_coef", type=float, default=0.001, help="KL divergence coefficient")
    parser.add_argument("--num_episodes", type=int, default=2000, help="Total episodes to train")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--model_checkpoint", type=str, default=None, help="Checkpoint to load")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--entropy_coef", type=float, default=2.0, help="Entropy coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--use_cuda", type=bool, default=True, help="Use CUDA if available")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel workers (default: cpu_count)")
    parser.add_argument("--ref_update_freq", type=int, default=10, help="Update reference model every N episodes")
    
    args = parser.parse_args()
    return args


def collect_trajectory_worker(args_tuple):
    """Worker function to collect a single trajectory in parallel"""
    rollout_idx, model_state_dict, device_str, opt_dict = args_tuple
    
    # Critical: Set SDL and threading before any imports
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    # Force CPU and disable CUDA in workers
    import torch
    torch.set_num_threads(1)
    
    import pygame
    pygame.init()
    from pygame.surfarray import array3d
    
    # Reconstruct options
    class OptNamespace:
        pass
    opt = OptNamespace()
    for key, value in opt_dict.items():
        setattr(opt, key, value)
    
    # Set device (use CPU for workers)
    device = torch.device('cpu')
    
    # Initialize model
    model = GRPOPolicyNetwork(device)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    
    # Direction mappings
    UP = 1
    DOWN = -1
    LEFT = 2
    RIGHT = -2
    direction_map = {0: UP, 1: DOWN, 2: LEFT, 3: RIGHT}
    
    # Initialize game
    game_state = Pacman()
    game_state.lives = 1
    game_state.lifesprites.resetLives(game_state.lives)
    game_state.score = 0
    game_state.startGame()
    prev_score = 0
    
    # Auto-press space to start
    event = pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_SPACE})
    pygame.event.post(event)
    
    # Get initial state
    frame = array3d(game_state.screen)
    image = pre_processing(frame, opt.image_size, opt.image_size)
    image = torch.from_numpy(image).to(device)
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]
    
    trajectory = []
    total_reward = 0
    steps_without_food = 0
    action_counts = [0, 0, 0, 0]
    pellets_eaten = 0
    
    # Track previous position
    prev_pacman_pos = None
    stuck_counter = 0
    position_history = []
    
    for t in range(opt.T_horizon):
        # Get action from policy
        with torch.no_grad():
            prob = model.pi(state)
        dist = Categorical(prob)
        
        action = dist.sample().item()
        action_prob = prob[0][action].item()
        action_counts[action] += 1
        
        # Take action in environment
        if game_state.pacman.alive:
            key = direction_map.get(action, None)
            if key is not None:
                game_state.pacman.next_direction = key
        
        # Store previous pacman position
        if game_state.pacman.alive:
            current_pos = (game_state.pacman.position.x, game_state.pacman.position.y)
            position_history.append(current_pos)
            if len(position_history) > 20:
                position_history.pop(0)
        
        # Update game
        game_state.update()
        
        # Calculate reward - IMPROVED REWARD SHAPING
        step_reward = game_state.score - prev_score
        reward = 0
        
        if step_reward > 0:
            pellets_eaten += 1
            if step_reward >= 50:  # Power pellet
                reward += 150.0
            else:  # Regular pellet
                reward += 30.0
            steps_without_food = 0
        else:
            reward += 0.0  # No penalty for exploration
            steps_without_food += 1
        
        # Distance-based reward (encourage moving toward pellets)
        if game_state.pacman.alive and hasattr(game_state, 'pellets'):
            try:
                pacman_pos = game_state.pacman.position
                min_dist = float('inf')
                for pellet in game_state.pellets.pelletList:
                    if pellet:
                        dist_to_pellet = ((pacman_pos.x - pellet.position.x)**2 + 
                                        (pacman_pos.y - pellet.position.y)**2)**0.5
                        min_dist = min(min_dist, dist_to_pellet)
                
                # Small reward for getting closer (encourage exploration)
                if min_dist < 100:
                    reward += 0.5
            except:
                pass
        
        # Check if stuck - REDUCED PENALTY
        if len(position_history) >= 20:
            unique_positions = len(set(position_history[-20:]))
            if unique_positions < 5:
                reward -= 0.5
                stuck_counter += 1
            else:
                stuck_counter = 0
        
        # Progressive penalty for not eating - LESS AGGRESSIVE
        if steps_without_food > 100:
            reward -= 0.2
        if steps_without_food > 200:
            reward -= 1.0
        
        prev_score = game_state.score
        
        # Check terminal state
        terminal = False
        if not game_state.pacman.alive:
            terminal = True
            if game_state.pellets.isEmpty():
                reward += 1000.0
            else:
                reward -= 100.0
        
        # Get next state
        frame = array3d(game_state.screen)
        next_image = pre_processing(frame, opt.image_size, opt.image_size)
        next_image = torch.from_numpy(next_image).to(device)
        next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]
        
        # Store transition
        trajectory.append({
            'state': state.cpu().numpy(),
            'action': action,
            'reward': reward,
            'next_state': next_state.cpu().numpy(),
            'action_prob': action_prob,
            'terminal': terminal
        })
        
        state = next_state
        total_reward += reward
        
        if terminal:
            break
    
    return {
        'trajectory': trajectory,
        'total_reward': total_reward,
        'terminal': terminal,
        'steps': t + 1,
        'final_score': prev_score,
        'pellets_eaten': pellets_eaten,
        'action_counts': action_counts,
        'stuck_counter': stuck_counter
    }


def compute_returns(rewards, gamma):
    """Compute discounted returns (Monte Carlo)"""
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def normalize_advantages_per_trajectory(all_trajectories, normalized_outcome_rewards, opt, device):
    """
    GRPO with per-trajectory return normalization + group outcome signal
    Combines step-level returns with group-relative advantages
    """
    tot_advantage_batch = []
    
    for traj_idx, trajectory in enumerate(all_trajectories):
        # Get trajectory rewards
        traj_rewards = [t['reward'] for t in trajectory]
        
        # Compute per-step returns (temporal credit assignment)
        returns = compute_returns(traj_rewards, opt.gamma)
        returns_tensor = torch.tensor(returns, dtype=torch.float)
        
        # Normalize returns within trajectory (reduces variance)
        if len(returns) > 1:
            returns_mean = returns_tensor.mean()
            returns_std = returns_tensor.std() + 1e-8
            normalized_returns = (returns_tensor - returns_mean) / returns_std
        else:
            normalized_returns = returns_tensor
        
        # Get group-relative outcome advantage
        group_advantage = normalized_outcome_rewards[traj_idx]
        
        # Combine: per-step returns + group outcome signal
        # This gives both temporal credit assignment AND group-relative comparison
        advantages = normalized_returns * (1.0 + 0.5 * group_advantage)
        
        tot_advantage_batch.append(advantages.unsqueeze(1).to(device))
    
    return tot_advantage_batch


def train(opt):
    # Set up device
    use_cuda = opt.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Determine number of workers
    if opt.num_workers is None:
        opt.num_workers = min(opt.num_rollouts, cpu_count())
    print(f"Using {opt.num_workers} parallel workers for {opt.num_rollouts} rollouts")
    
    # Set seeds
    if use_cuda:
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    
    # Initialize models
    model = GRPOPolicyNetwork(device)
    model.to(device)
    
    # Reference model (frozen, updated periodically)
    ref_model = GRPOPolicyNetwork(device)
    ref_model.to(device)
    
    # Load checkpoint if specified
    start_episode = 0
    running_reward = 0
    global_action_counts = [0, 0, 0, 0]
    best_reward = float('-inf')
    
    if opt.model_checkpoint:
        checkpoint_path = os.path.join(opt.saved_path, opt.model_checkpoint)
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_episode = checkpoint.get('episode', 0) + 1
            running_reward = checkpoint.get('running_reward', 0)
            best_reward = checkpoint.get('best_reward', float('-inf'))
            global_action_counts = checkpoint.get('action_counts', [0, 0, 0, 0])
            print(f"Resuming from episode {start_episode}")
    
    ref_model.load_state_dict(model.state_dict())
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-4)
    
    # Learning rate scheduler - ADDED
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    
    # Load optimizer state if available
    if opt.model_checkpoint:
        checkpoint_path = os.path.join(opt.saved_path, opt.model_checkpoint)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print(f"Loaded optimizer state from checkpoint")
                except Exception as e:
                    print(f"Could not load optimizer state: {e}")
            
            # Load scheduler state if available
            if 'scheduler_state_dict' in checkpoint:
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    print(f"Loaded scheduler state from checkpoint")
                except Exception as e:
                    print(f"Could not load scheduler state: {e}")
            
            # Force LR to command line argument value
            for param_group in optimizer.param_groups:
                param_group['lr'] = opt.lr
            
            print(f"Set learning rate to: {opt.lr}")
    
    # Verify learning rate
    current_lr = optimizer.param_groups[0]['lr']
    assert current_lr == opt.lr, f"Learning rate mismatch! Expected {opt.lr}, got {current_lr}"
    
    # Tensorboard
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path, exist_ok=True)
    writer = SummaryWriter(opt.log_path)
    
    # Log hyperparameters
    writer.add_text('Hyperparameters', 
                    f"lr={opt.lr}, gamma={opt.gamma}, "
                    f"eps_clip={opt.eps_clip}, K_epoch={opt.K_epoch}, "
                    f"num_rollouts={opt.num_rollouts}, num_workers={opt.num_workers}, "
                    f"entropy_coef={opt.entropy_coef}, kl_coef={opt.kl_coef}, "
                    f"ref_update_freq={opt.ref_update_freq}", 0)
    
    print(f"TensorBoard logging to: {opt.log_path}")
    print(f"Run 'tensorboard --logdir={opt.log_path}' to view")
    
    # Training loop
    os.makedirs(opt.saved_path, exist_ok=True)
    n_epi = start_episode
    score = 0.0
    scores = []
    print_interval = 20
    
    pbar = tqdm(total=opt.num_episodes, initial=start_episode, desc="Episodes")
    
    # Create process pool
    from multiprocessing import get_context
    ctx = get_context("spawn")
    pool = ctx.Pool(processes=opt.num_workers)
    
    try:
        while n_epi < start_episode + opt.num_episodes:
            # Prepare arguments for parallel trajectory collection
            opt_dict = vars(opt)
            model_state_dict = ref_model.state_dict()
            
            # Move state dict to CPU for serialization
            model_state_dict_cpu = {k: v.cpu() for k, v in model_state_dict.items()}
            
            args_list = [
                (i, model_state_dict_cpu, 'cpu', opt_dict)
                for i in range(opt.num_rollouts)
            ]
            
            # Collect trajectories in parallel
            try:
                results = pool.map_async(collect_trajectory_worker, args_list).get(timeout=300)
            except Exception as e:
                print(f"\nError collecting trajectories: {e}")
                print("Retrying with sequential collection...")
                results = []
                for args in args_list[:1]:
                    results.append(collect_trajectory_worker(args))
            
            # Process results and compute group-relative advantages
            all_trajectories = []
            all_rewards = []
            all_terminals = []
            all_steps = []
            all_pellets = []
            
            # Group-level statistics for GRPO
            group_outcome_rewards = []
            
            for result in results:
                all_trajectories.append(result['trajectory'])
                all_rewards.append(result['total_reward'])
                all_terminals.append(result['terminal'])
                all_steps.append(result['steps'])
                all_pellets.append(result['pellets_eaten'])
                
                # Compute trajectory return (outcome reward)
                trajectory_rewards = [t['reward'] for t in result['trajectory']]
                trajectory_return = sum([r * (opt.gamma ** i) for i, r in enumerate(trajectory_rewards)])
                group_outcome_rewards.append(trajectory_return)
                
                # Update global action counts
                for i, count in enumerate(result['action_counts']):
                    global_action_counts[i] += count
            
            # Update episode counter
            n_epi += 1
            pbar.update(1)
            
            episode_reward = all_rewards[0]
            score += episode_reward
            scores.append(episode_reward)
            running_reward = 0.05 * episode_reward + 0.95 * running_reward
            
            if episode_reward > best_reward:
                best_reward = episode_reward
            
            # GRPO: Normalize outcome rewards within the group
            group_outcome_rewards_np = np.array(group_outcome_rewards)
            mean_reward = np.mean(group_outcome_rewards_np)
            std_reward = np.std(group_outcome_rewards_np) + 1e-8
            normalized_outcome_rewards = (group_outcome_rewards_np - mean_reward) / std_reward
            
            # Clamp advantages to prevent extreme values
            normalized_outcome_rewards = np.clip(normalized_outcome_rewards, -10.0, 10.0)
            
            # Prepare batches for training
            tot_s_batch = []
            tot_a_batch = []
            tot_prob_batch = []
            
            for traj_idx, trajectory in enumerate(all_trajectories):
                states = []
                actions = []
                probs = []
                
                for transition in trajectory:
                    s = torch.from_numpy(transition['state']).to(device)
                    states.append(s)
                    actions.append(transition['action'])
                    probs.append(transition['action_prob'])
                
                s_batch = torch.cat(states)
                a_batch = torch.tensor([[a] for a in actions]).to(device)
                prob_batch = torch.tensor([[p] for p in probs], dtype=torch.float).to(device)
                
                tot_s_batch.append(s_batch)
                tot_a_batch.append(a_batch)
                tot_prob_batch.append(prob_batch)
            
            # IMPROVED: Compute advantages with per-trajectory normalization + group signal
            tot_advantage_batch = normalize_advantages_per_trajectory(
                all_trajectories, normalized_outcome_rewards, opt, device
            )
            
            # Update policy with GRPO
            model.train()
            total_policy_loss = 0
            total_entropy = 0
            total_kl_div = 0
            num_batches = 0
            
            for epoch in range(opt.K_epoch):
                epoch_loss = 0
                
                for s_batch, a_batch, prob_batch, adv_batch in zip(
                    tot_s_batch, tot_a_batch, tot_prob_batch, tot_advantage_batch
                ):
                    # Get current policy
                    pi = model.pi(s_batch)
                    dist = Categorical(pi)
                    
                    # Get reference policy (frozen)
                    with torch.no_grad():
                        ref_pi = ref_model.pi(s_batch)
                    
                    # Compute log probabilities
                    log_pi_a = dist.log_prob(a_batch.squeeze()).unsqueeze(1)
                    log_prob_a = torch.log(prob_batch + 1e-8)
                    
                    # Importance sampling ratio
                    ratio = torch.exp(log_pi_a - log_prob_a)
                    
                    # PPO clipped objective
                    surr1 = ratio * adv_batch
                    surr2 = torch.clamp(ratio, 1 - opt.eps_clip, 1 + opt.eps_clip) * adv_batch
                    policy_loss = -torch.min(surr1, surr2)
                    
                    # Entropy bonus
                    entropy = dist.entropy().unsqueeze(1)
                    
                    # KL divergence (using unbiased estimator from DeepSeek paper)
                    pi_a = pi.gather(1, a_batch)
                    ref_pi_a = ref_pi.gather(1, a_batch)
                    
                    ratio_kl = ref_pi_a / (pi_a + 1e-8)
                    kl_div = ratio_kl - torch.log(ratio_kl + 1e-8) - 1
                    
                    # Total loss (GRPO objective from Equation 3 in paper)
                    loss = (policy_loss.mean() - 
                           opt.entropy_coef * entropy.mean() +
                           opt.kl_coef * kl_div.mean()) / opt.gradient_accumulation_steps
                    
                    epoch_loss += loss
                    
                    # Accumulate metrics
                    total_policy_loss += policy_loss.mean().item()
                    total_entropy += entropy.mean().item()
                    total_kl_div += kl_div.mean().item()
                    num_batches += 1
                
                # Backward pass
                if torch.isnan(epoch_loss).any():
                    print("Warning: NaN detected. Skipping backward pass.")
                    optimizer.zero_grad()
                else:
                    epoch_loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
            
            # Step the learning rate scheduler
            scheduler.step()
            
            # Compute average metrics
            avg_policy_loss = total_policy_loss / num_batches if num_batches > 0 else 0
            avg_entropy = total_entropy / num_batches if num_batches > 0 else 0
            avg_kl_div = total_kl_div / num_batches if num_batches > 0 else 0
            
            # Update reference model periodically
            if n_epi % opt.ref_update_freq == 0:
                ref_model.load_state_dict(model.state_dict())
            
            # Logging
            writer.add_scalar('Train/EpisodeReward', episode_reward, n_epi)
            writer.add_scalar('Train/RunningReward', running_reward, n_epi)
            writer.add_scalar('Train/BestReward', best_reward, n_epi)
            writer.add_scalar('Train/EpisodeLength', all_steps[0], n_epi)
            writer.add_scalar('Train/PelletsEaten', all_pellets[0], n_epi)
            writer.add_scalar('Train/PolicyLoss', avg_policy_loss, n_epi)
            writer.add_scalar('Train/Entropy', avg_entropy, n_epi)
            writer.add_scalar('Train/KL_Divergence', avg_kl_div, n_epi)
            writer.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], n_epi)
            writer.add_scalar('Train/AvgGroupReward', mean_reward, n_epi)
            writer.add_scalar('Train/StdGroupReward', std_reward, n_epi)
            
            # Log action distribution
            total_actions = sum(global_action_counts)
            if total_actions > 0:
                for i, count in enumerate(global_action_counts):
                    writer.add_scalar(f'Train/ActionDist/Action_{i}', count / total_actions, n_epi)
            
            action_dist = [a/total_actions for a in global_action_counts] if total_actions > 0 else [0.25]*4
            
            pbar.set_postfix({
                'Reward': f'{episode_reward:.2f}',
                'Running': f'{running_reward:.2f}',
                'Best': f'{best_reward:.2f}',
                'Pellets': f'{all_pellets[0]}',
            })
            
            # Print statistics
            if n_epi % print_interval == 0 and n_epi != 0:
                avg_score = score / print_interval
                avg_pellets = sum(all_pellets) / len(all_pellets)
                print(f"\n{'='*70}")
                print(f"Episode: {n_epi}")
                print(f"  Avg Reward: {avg_score:.2f} | Running: {running_reward:.2f} | Best: {best_reward:.2f}")
                print(f"  Avg Pellets: {avg_pellets:.1f} | Avg Steps: {sum(all_steps)/len(all_steps):.1f}")
                print(f"  Entropy: {avg_entropy:.3f} | KL Div: {avg_kl_div:.3f}")
                print(f"  Group Mean Reward: {mean_reward:.2f} | Group Std: {std_reward:.2f}")
                print(f"  Action Dist: UP={action_dist[0]:.2f} DOWN={action_dist[1]:.2f} LEFT={action_dist[2]:.2f} RIGHT={action_dist[3]:.2f}")
                print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
                print(f"{'='*70}")
                score = 0.0
            
            # Save checkpoint
            if n_epi % 250 == 0 and n_epi != 0:
                checkpoint_data = {
                    'episode': n_epi,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'running_reward': running_reward,
                    'best_reward': best_reward,
                    'action_counts': global_action_counts
                }
                torch.save(checkpoint_data, f"{opt.saved_path}/pacman_{n_epi}.pth")
                torch.save(checkpoint_data, f"{opt.saved_path}/pacman_latest.pth")
            
            # Save best model
            if episode_reward >= best_reward * 0.95:
                checkpoint_data = {
                    'episode': n_epi,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'running_reward': running_reward,
                    'best_reward': best_reward,
                    'action_counts': global_action_counts
                }
                torch.save(checkpoint_data, f"{opt.saved_path}/pacman_best.pth")
            
            # Cleanup
            del tot_s_batch, tot_a_batch, tot_prob_batch, tot_advantage_batch
            del all_trajectories
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    finally:
        pool.close()
        pool.join()
    
    pbar.close()
    
    # Flush and close writer
    writer.flush()
    writer.close()
    
    # Save final model
    torch.save({
        'episode': n_epi,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'running_reward': running_reward,
        'best_reward': best_reward,
        'action_counts': global_action_counts
    }, f"{opt.saved_path}/pacman_final.pth")
    
    print("\nTraining completed!")
    print(f"Best reward achieved: {best_reward:.2f}")
    print(f"TensorBoard logs saved to: {opt.log_path}")
    print(f"To view: tensorboard --logdir={opt.log_path}")


if __name__ == "__main__":
    opt = get_args()
    train(opt)