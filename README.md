# ![Pac-Man Icon](https://upload.wikimedia.org/wikipedia/commons/4/49/Pacman.svg) GRPO for Pacman

> **⚠️ IMPORTANT ATTRIBUTION**: This project uses the Pacman game environment from [pacmancode.com](http://pacmancode.com). The Pacman game code is **not included in this repository** and must be obtained separately from the original source. Please respect the original author's work and licensing terms.

A PyTorch implementation of **Group Relative Policy Optimization (GRPO)** for training an agent to play Pacman. GRPO is a reinforcement learning algorithm that optimizes policies by comparing outcomes within groups of rollouts, combining the benefits of PPO with group-relative advantage estimation.

## 🌟 Features

- **GRPO Algorithm**: Implementation based on DeepSeek's paper with group-relative advantage normalization
- **Parallel Trajectory Collection**: Multi-process rollout collection for efficient training
- **Advanced Reward Shaping**: Distance-based rewards and exploration bonuses
- **Valid Action Masking**: Prevents the agent from attempting invalid moves (wall collisions)
- **Comprehensive Logging**: TensorBoard integration with detailed metrics
- **Checkpoint Management**: Automatic saving of best models and periodic checkpoints
- **Testing Framework**: Evaluate trained models with various sampling strategies

## 📚 Acknowledgments

This implementation was inspired by and builds upon concepts from:
- [From LLMs to Flappy Bird: Applying DeepSeek's GRPO to Game AI](https://medium.com/@psopen11/from-llms-to-flappy-bird-applying-deepseeks-grpo-to-game-ai-8e1de1056957) - Excellent article explaining GRPO adaptation for game environments
- DeepSeek's GRPO paper for the core algorithm
- [pacmancode.com](http://pacmancode.com) for the Pacman game environment

## 🛠️ Installation

### Prerequisites

- Python 3.12 (tested and recommended)
- CUDA-capable GPU (recommended, but CPU training is supported)

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd grpo-pacman
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

The required packages are:
- `numpy` - Numerical computing
- `torch` - PyTorch deep learning framework
- `tensorboardX` - TensorBoard logging
- `pygame` - Game rendering and environment
- `opencv-python` - Image preprocessing
- `tqdm` - Progress bars

### Step 3: Obtain Pacman Environment

**CRITICAL**: You must obtain the Pacman game code separately from [pacmancode.com](http://pacmancode.com).

1. Visit [pacmancode.com](http://pacmancode.com) and download the Pacman game code
2. Extract the Pacman code into `src/Pacman_Complete/`
3. Your directory structure should look like:
   ```
   grpo-pacman/
   ├── src/
   │   ├── Pacman_Complete/
   │   │   ├── run.py
   │   │   ├── pellets.py
   │   │   └── ... (other Pacman files)
   │   ├── grpo_network.py
   │   └── utils.py
   ├── train.py
   ├── test.py
   └── requirements.txt
   ```
4. imports must be fixed and modifed to work with codebase

### Step 4: Verify Installation

Test that everything is set up correctly:

```bash
python -c "import torch; import pygame; print('Installation successful!')"
```

## 🚀 Usage

### Training

Basic training with default parameters:

```bash
python train.py
```

Advanced training with custom parameters:

```bash
python train.py \
    --num_episodes 2000 \
    --batch_size 50 \
    --lr 1e-4 \
    --gamma 0.9 \
    --num_rollouts 8 \
    --num_workers 4 \
    --entropy_coef 2.0 \
    --kl_coef 0.001 \
    --saved_path trained_models \
    --log_path tensorboard
```

Resume training from a checkpoint:

```bash
python train.py --model_checkpoint pacman_latest.pth
```

#### Key Training Parameters

- `--num_episodes`: Total training episodes (default: 2000)
- `--batch_size`: Samples per mini-batch (default: 50)
- `--lr`: Learning rate (default: 1e-4)
- `--gamma`: Discount factor (default: 0.9)
- `--num_rollouts`: Rollouts per group for GRPO (default: 8)
- `--num_workers`: Parallel workers for trajectory collection (default: CPU count)
- `--entropy_coef`: Entropy bonus coefficient (default: 2.0)
- `--kl_coef`: KL divergence penalty (default: 0.001)
- `--eps_clip`: PPO clipping parameter (default: 0.2)
- `--K_epoch`: Policy update epochs per iteration (default: 3)

### Testing

Evaluate a trained model:

```bash
python test.py --model_name pacman_best.pth --num_episodes 10
```

Test with different sampling strategies:

```bash
# Deterministic policy (argmax)
python test.py --model_name pacman_best.pth --deterministic

# Epsilon-greedy exploration
python test.py --model_name pacman_best.pth --epsilon 0.1

# Temperature-based sampling
python test.py --model_name pacman_best.pth --temperature 0.8
```

#### Testing Parameters

- `--model_name`: Model checkpoint to load (default: pacman_final.pth)
- `--num_episodes`: Number of test episodes (default: 10)
- `--deterministic`: Use argmax action selection
- `--epsilon`: Epsilon for ε-greedy exploration (default: 0.0)
- `--temperature`: Softmax temperature for sampling (default: 1.0)
- `--headless`: Run without display

### Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir=tensorboard
```

Then open http://localhost:6006 in your browser.

Tracked metrics include:
- Episode reward and running average
- Policy loss and entropy
- KL divergence from reference policy
- Action distribution
- Pellets eaten per episode
- Learning rate schedule

## 🧠 Algorithm Details

### GRPO Overview

Group Relative Policy Optimization (GRPO) improves upon PPO by:

1. **Group-based Rollouts**: Collecting multiple trajectories (rollouts) in parallel
2. **Relative Advantage**: Computing advantages relative to the mean within each group
3. **Outcome Normalization**: Normalizing trajectory returns for variance reduction
4. **KL Regularization**: Penalizing divergence from a reference policy

### Key Components

**Policy Network** (`grpo_network.py`):
- Convolutional feature extraction (3 conv layers with batch norm)
- Fully connected layers with dropout
- Policy head outputting action probabilities (no value head needed for GRPO)

**Reward Shaping**:
- +30 points for regular pellets
- +150 points for power pellets
- +1000 points for winning
- -100 points for dying
- Small distance-based rewards for exploration
- Penalties for getting stuck

**Valid Action Masking**:
- Filters out actions that lead into walls
- Prevents wasted exploration on invalid moves
- Uses Pacman's built-in direction validation

## 📊 Training Progress

Typical training progression:
- **Episodes 0-500**: Agent learns basic movement and pellet collection
- **Episodes 500-1000**: Improved exploration and fewer deaths
- **Episodes 1000-1500**: Strategic pellet collection patterns emerge
- **Episodes 1500-2000**: Refined policy with consistent performance

Expected performance:
- Average score: 500-800 points
- Pellets eaten: 50-80% of total
- Episode length: 500-1500 steps

## 🐛 Troubleshooting

### Common Issues

**Import errors with Pacman code:**
- Ensure Pacman code is in `src/Pacman_Complete/`
- Verify all Pacman dependencies are satisfied
- Check that `run.py` and `pellets.py` are present

**CUDA out of memory:**
```bash
# Reduce batch size and number of rollouts
python train.py --batch_size 32 --num_rollouts 4
```

**Training hangs or freezes:**
```bash
# Reduce number of workers
python train.py --num_workers 2
```

**Poor performance:**
- Try adjusting entropy coefficient for more exploration
- Increase learning rate initially, then decay
- Check TensorBoard for signs of instability (NaN values, exploding gradients)

## 📁 Project Structure

```
grpo-pacman/
├── src/
│   ├── Pacman_Complete/      # Pacman game environment (not included)
│   ├── grpo_network.py        # GRPO policy network architecture
│   └── utils.py               # Image preprocessing utilities
├── trained_models/            # Saved model checkpoints
├── tensorboard/               # TensorBoard logs
├── train.py                   # Training script
├── test.py                    # Testing/evaluation script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

Areas for improvement:
- Hyperparameter tuning
- Alternative network architectures
- Additional reward shaping strategies
- Multi-agent scenarios (ghosts as adversaries)

## 📄 License

This project is licensed under the MIT License - see below for details.

**Important**: The Pacman game environment from [pacmancode.com](http://pacmancode.com) has its own separate license and is **not included** in this repository. You must obtain it separately and comply with its licensing terms.

## 📖 References

- DeepSeek GRPO Paper: Group Relative Policy Optimization
- [From LLMs to Flappy Bird: Applying DeepSeek's GRPO to Game AI](https://medium.com/@psopen11/from-llms-to-flappy-bird-applying-deepseeks-grpo-to-game-ai-8e1de1056957)
- [Pacman Code](http://pacmancode.com) - Original Pacman implementation
- Schulman et al. (2017): Proximal Policy Optimization Algorithms

## 💬 Citation

If you use this code in your research, please cite:

```bibtex
@misc{grpo-pacman,
  author = {Your Name},
  title = {GRPO for Pacman},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/grpo-pacman}
}
```

---

**Happy Training! 🎮🤖**