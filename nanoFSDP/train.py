import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchvision import datasets, transforms

from model import ShardedCNN, all_reduce

def setup_dataset(world_size, rank, bs=32):
    """Setup the dataloader with Distributed Sampling"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(
        root='data/', 
        train=True, 
        transform=transform, 
        download=True
    )
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank
    )

    dataloader = DataLoader(
        dataset,
        batch_size=bs,
        sampler=sampler
    )

    return dataloader

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

    dataloader = setup_dataset(world_size, rank, bs=64)

    # Run training for 100 epochs
    for epoch in range(100):
        running_loss = 0
        # Transfer the input data to the current rank
        for idx, (images, targets) in enumerate(dataloader):
            inputs = images.cuda(rank)
            targets = targets.cuda(rank)

            # Clear out the optimizer
            optimizer.zero_grad()
            
            # Run one forward pass and calculate loss
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            
            # Backprop the loss and sync the gradients
            loss.backward()
            all_reduce(model)
            running_loss += loss.item()

            # Take one optimizer step
            optimizer.step()

            if idx > 0 and idx % 100 == 0 and rank == 0:
                print(f"epoch: {epoch+1}, batch: {idx+1}, loss: {running_loss / 100:.4f}")
                running_loss = 0.0
    
    dist.destroy_process_group()

if __name__ == "__main__":
    # Run the training
    world_size = 2
    mp.spawn(train_fsdp, args=(world_size,), nprocs=world_size, join=True)