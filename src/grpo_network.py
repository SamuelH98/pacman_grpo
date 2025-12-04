"""
GRPO Policy Network for Pacman
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GRPOPolicyNetwork(nn.Module):
    def __init__(self, device):
        super(GRPOPolicyNetwork, self).__init__()
        self.device = device

        # Convolutional layers with batch normalization
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Feature extraction with dropout
        self.features = nn.Sequential(
            nn.Linear(7 * 7 * 128, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # Policy head (actor) - ONLY component needed for GRPO
        # Outputs action logits for 4 actions: UP, DOWN, LEFT, RIGHT
        self.actor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4)
        )
        
        self._create_weights()

    def _create_weights(self):
        """Initialize weights with orthogonal initialization for policy stability"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Special initialization for actor output layer (smaller values for stable policy)
        if hasattr(self.actor[-1], 'weight'):
            nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
            nn.init.constant_(self.actor[-1].bias, 0)

    def forward(self, x):
        """Extract features from input state"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.features(x)
        return x

    def pi(self, x, softmax_dim=1):
        """
        Get action probabilities from the policy network.
        
        This is the ONLY output needed for GRPO!
        No value function estimation required.
        
        Args:
            x: Input state tensor
            softmax_dim: Dimension for softmax (default: 1)
            
        Returns:
            Action probabilities (clamped for numerical stability)
        """
        features = self.forward(x)
        logits = self.actor(features)
        prob = F.softmax(logits, dim=softmax_dim)
        # Clamp to prevent numerical instability
        prob = torch.clamp(prob, 1e-8, 1.0 - 1e-8)
        return prob
    
    def get_action(self, x, deterministic=False):
        """
        Sample an action from the policy.
        
        Args:
            x: Input state tensor
            deterministic: If True, return argmax action; if False, sample
            
        Returns:
            action: Selected action (int)
            log_prob: Log probability of the action
            prob: Probability distribution over actions
        """
        prob = self.pi(x)
        
        if deterministic:
            action = torch.argmax(prob, dim=1)
            log_prob = torch.log(prob.gather(1, action.unsqueeze(1)))
        else:
            from torch.distributions import Categorical
            dist = Categorical(prob)
            action = dist.sample()
            log_prob = dist.log_prob(action).unsqueeze(1)
        
        return action.item() if action.dim() == 0 else action, log_prob, prob