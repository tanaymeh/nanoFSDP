import torch
import torch.nn as nn
import torch.distributed as dist

def reset_shards(cls) -> None:
    # Resets the shards back to None, optionally for Bias too if it's enabled
    cls.weight_shard = None
    cls.weight_grad_shard = None

    if cls.bias is not None:
        cls.bias_shard = None
        cls.bias_grad_shard = None


class ShardedConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reset_shards(self)

    def shard_parameters(self, rank, world_size):
        # Get the current shard's size
        # Since each channel is a different feature map, dividing by num gpus will determine
        # how many feature maps (which is a shard) go to each GPU.
        shard_size = self.out_channels // world_size
        
        # Starting index of the self shard
        start_idx = rank * shard_size
        end_idx = self.out_channels if not rank < world_size - 1 else start_idx + shard_size

        # Shard the weights
        self.weight_shard = self.weight[start_idx:end_idx].clone()
        self.weight_grad = torch.zeros_like(self.weight_shard)

        # Shard the bias
        if self.bias is not None:
            self.bias_shard = self.bias[start_idx:end_idx].clone()
            self.bias_grad = torch.zeros_like(self.bias_shard)
    
    def forward(self, x):
        # Perform the computation on the current device with current shard
        local_shard_output = nn.functional.conv1d(
            x, 
            self.weight_shard, 
            self.bias_shard, 
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )
        # Init an empty tensor list with the shape of the final gathered output tensor
        gathered_output = [torch.zeros_like(local_shard_output) for _ in range(dist.get_world_size())]

        # Gather the partial computed result from all GPUs and convert them in a single result tensor
        dist.all_gather(gathered_output, local_shard_output)
        return torch.cat(gathered_output, dim=1)


class ShardedLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reset_shards(self)

    def shard_parameters(self, rank, world_size):
        # Get the current shard's size
        # Each shard in this case will have out_features // world_size number of features in it
        shard_size = self.out_features // world_size
        
        # Starting index of the self shard
        start_idx = rank * shard_size
        end_idx = self.out_channels if not rank < world_size - 1 else start_idx + shard_size

        # Shard the weights
        self.weight_shard = self.weight[start_idx:end_idx].clone()
        self.weight_grad = torch.zeros_like(self.weight_shard)

        # Shard the bias
        if self.bias is not None:
            self.bias_shard = self.bias[start_idx:end_idx].clone()
            self.bias_grad = torch.zeros_like(self.bias_shard)

    def forward(self, x):
        # Perform the computation on the current device with current shard
        local_shard_output = nn.functional.linear(
            x, 
            self.weight_shard, 
            self.bias_shard, 
        )
        # Init an empty tensor list with the shape of the final gathered output tensor
        gathered_output = [torch.zeros_like(local_shard_output) for _ in range(dist.get_world_size())]

        # Gather the partial computed result from all GPUs and convert them in a single result tensor
        dist.all_gather(gathered_output, local_shard_output)
        return torch.cat(gathered_output, dim=1)