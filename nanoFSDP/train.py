import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

from model import ShardedCNN, all_reduce

def train_fsdp(rank, world_size):
    """Trains a model using FSDP training strategy"""
    # Important to use 'nccl' instead of 'gloo' because it's nccl is faster on GPU-to-GPU comms
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Init the model and shard the parameters
    model = ShardedCNN(num_classes=10)
    model.shard_parameters(rank, world_size)

    # Init optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Run training for 100 epochs
    for epoch in range(100):
        # Transfer the input data to the current rank
        inputs = torch.randn(64, 1, 128).cuda(rank)
        targets = torch.randint(0, 10, (64,)).cuda(rank)

        # Clear out the optimizer
        optimizer.zero_grad()
        
        # Run one forward pass and calculate loss
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        
        # Backprop the loss and sync the gradients
        loss.backward()
        all_reduce(model)

        # Take one optimizer step
        optimizer.step()

        if rank == 0:
            print(f"epoch: {epoch} / 100, train_loss: {loss.item():.4f}")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    # Run the training
    world_size = 2
    mp.spawn(train_fsdp, args=(world_size,), nprocs=world_size, join=True)