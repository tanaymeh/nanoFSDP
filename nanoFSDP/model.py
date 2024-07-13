import torch
import torch.nn as nn
import torch.distributed as dist

from modules import ShardedLinear, ShardedConv1d

class ShardedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = ShardedConv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = ShardedConv1d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.AvgPool1d(2)
        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.2)

        self.fc1 = ShardedLinear(64 * 32, 128)
        self.fc2 = ShardedLinear(128, num_classes)

    def shard_parameters(self, rank, world_size):
        self.conv1.shard_parameters(rank, world_size)
        self.conv2.shard_parameters(rank, world_size)
        self.fc1.shard_parameters(rank, world_size)
        self.fc2.shard_parameters(rank, world_size)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dro1(x)
        x = x.view(x.size(0), -1)
        
        x = torch.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)

        return x
    
def all_reduce(model):
    """Synchronises gradients across all ranks by running all_reduce on all parameters"""
    for param in model.parameters():
        if param.grad is not None:
            # Gather the element-wise sum of the distributed gradients using all_reduce
            dist.all_reduce(param.grad)
            # Average them
            param.grad /= dist.get_world_size()

