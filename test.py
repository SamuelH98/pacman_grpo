"""
GRPO Testing for Pacman - Fixed Wall Collision
"""
import argparse
import sys
import os
import pygame
from pygame.surfarray import array3d
import torch
from torch.distributions import Categorical
from tqdm import tqdm

from src.grpo_network import GRPOPolicyNetwork
from src.Pacman_Complete.run import GameController as Pacman
from src.Pacman_Complete.pellets import PelletGroup
from src.utils import pre_processing


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of GRPO to play Pacman - Testing""")
    parser.add_argument("--image_size", type=int, default=84, help="The common width and height for all images")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--model_name", type=str, default="pacman_final.pth", help="Name of model file to load")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of test episodes")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no display)")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy (argmax instead of sampling)")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Epsilon for epsilon-greedy exploration")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for softmax sampling")
    
    args = parser.parse_args()
    return args


def get_valid_actions(game_state):
    """
    Returns a mask of valid actions (actions that don't lead into walls)
    Returns: list of valid action indices [0=UP, 1=DOWN, 2=LEFT, 3=RIGHT]
    """
    UP = 1
    DOWN = -1
    LEFT = 2
    RIGHT = -2
    
    pacman = game_state.pacman
    valid_actions = []
    
    # Check each direction
    directions = [UP, DOWN, LEFT, RIGHT]
    for i, direction in enumerate(directions):
        # Use the game's built-in valid directions check (it's a method, so call it)
        if direction in pacman.validDirections():
            valid_actions.append(i)
    
    # If no valid actions (shouldn't happen), allow all
    if len(valid_actions) == 0:
        valid_actions = [0, 1, 2, 3]
    
    return valid_actions


def select_action_with_masking(prob, valid_actions, deterministic=False, epsilon=0.0, temperature=1.0):
    """
    Select action with valid action masking
    """
    # Create action mask
    mask = torch.zeros_like(prob[0])
    mask[valid_actions] = 1.0
    
    # Apply mask and renormalize
    masked_prob = prob[0] * mask
    masked_prob = masked_prob / (masked_prob.sum() + 1e-8)
    
    # Epsilon-greedy exploration
    if epsilon > 0 and torch.rand(1).item() < epsilon:
        # Random action from valid actions
        return valid_actions[torch.randint(len(valid_actions), (1,)).item()]
    
    if deterministic:
        # Argmax over valid actions only
        return valid_actions[masked_prob[valid_actions].argmax().item()]
    else:
        # Apply temperature for sampling
        if temperature != 1.0:
            masked_prob = torch.pow(masked_prob, 1.0 / temperature)
            masked_prob = masked_prob / masked_prob.sum()
        
        # Sample from masked distribution
        dist = Categorical(masked_prob)
        return dist.sample().item()


def test(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model_path = f"{opt.saved_path}/{opt.model_name}"
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model
    model = GRPOPolicyNetwork(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully from episode {checkpoint.get('episode', 'unknown')}")
    print(f"Policy Mode: {'Deterministic (argmax)' if opt.deterministic else 'Stochastic (sampling)'}")
    if opt.epsilon > 0:
        print(f"Epsilon-greedy: {opt.epsilon}")
    if opt.temperature != 1.0:
        print(f"Temperature: {opt.temperature}")

    # Initialize game
    game_state = Pacman()
    
    UP = 1
    DOWN = -1
    LEFT = 2
    RIGHT = -2
    direction_map = {0: UP, 1: DOWN, 2: LEFT, 3: RIGHT}

    episode_scores = []
    episode_steps_list = []
    pellets_eaten_list = []
    
    pbar = tqdm(total=opt.num_episodes, desc="Testing", unit="episode")

    for episode in range(opt.num_episodes):
        # Reset game
        game_state.lives = 1
        game_state.lifesprites.resetLives(game_state.lives)
        game_state.score = 0
        game_state.startGame()
        prev_score = game_state.score
        initial_pellets = len(game_state.pellets.pelletList)

        # Capture initial frame
        frame = array3d(game_state.screen)
        image = pre_processing(frame, opt.image_size, opt.image_size)
        image = torch.from_numpy(image)
        if torch.cuda.is_available():
            image = image.cuda()
        state = torch.cat([image for _ in range(4)])[None, :, :, :]

        episode_score = 0
        episode_steps = 0
        max_steps = 100000  # Prevent infinite episodes

        # Auto-press space to start
        event = pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_SPACE})
        pygame.event.post(event)

        while episode_steps < max_steps:
            with torch.no_grad():
                # Get action probabilities from policy
                prob = model.pi(state)
                
                # Get valid actions (not blocked by walls)
                valid_actions = get_valid_actions(game_state)
                
                # Select action with masking
                action = select_action_with_masking(
                    prob, 
                    valid_actions, 
                    deterministic=opt.deterministic,
                    epsilon=opt.epsilon,
                    temperature=opt.temperature
                )

            # Take action
            if game_state.pacman.alive:
                key = direction_map.get(action, None)
                if key is not None:
                    game_state.pacman.next_direction = key

            # Update game
            game_state.update()

            # Track score
            if game_state.score > prev_score:
                episode_score += (game_state.score - prev_score)
                prev_score = game_state.score

            # Check if Pacman died
            if not game_state.pacman.alive:
                break

            # Check if won (all pellets eaten)
            if game_state.pellets.isEmpty():
                print(f"  Episode {episode+1}: WON! All pellets cleared!")
                break

            # Get next frame
            frame = array3d(game_state.screen)
            next_image = pre_processing(frame, opt.image_size, opt.image_size)
            next_image = torch.from_numpy(next_image)
            if torch.cuda.is_available():
                next_image = next_image.cuda()
            next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]

            state = next_state
            episode_steps += 1

        # Calculate pellets eaten
        pellets_remaining = len(game_state.pellets.pelletList)
        pellets_eaten = initial_pellets - pellets_remaining
        
        episode_scores.append(episode_score)
        episode_steps_list.append(episode_steps)
        pellets_eaten_list.append(pellets_eaten)
        
        pbar.update(1)
        pbar.set_postfix({
            'Score': episode_score, 
            'Steps': episode_steps,
            'Pellets': f'{pellets_eaten}/{initial_pellets}'
        })

    pbar.close()

    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"Test Results ({opt.num_episodes} episodes)")
    print(f"{'='*60}")
    print(f"Average Score: {sum(episode_scores) / len(episode_scores):.2f}")
    print(f"Max Score: {max(episode_scores)}")
    print(f"Min Score: {min(episode_scores)}")
    print(f"Average Steps: {sum(episode_steps_list) / len(episode_steps_list):.2f}")
    print(f"Average Pellets Eaten: {sum(pellets_eaten_list) / len(pellets_eaten_list):.2f}/{initial_pellets}")
    print(f"{'='*60}")


if __name__ == "__main__":
    opt = get_args()
    test(opt)