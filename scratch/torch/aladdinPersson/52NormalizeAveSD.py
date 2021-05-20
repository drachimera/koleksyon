import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Load data
cifar10_train = datasets.CIFAR10(root='dataset/', train=True,
                                transform=transforms.ToTensor(), download=True)
cifar10_loader = DataLoader(dataset=cifar10_train, batch_size=64, shuffle=True, drop_last=True)

mnist_train = datasets.MNIST(root='dataset/', train=True,
                                transform=transforms.ToTensor(), download=True)
mnist_loader = DataLoader(dataset=mnist_train, batch_size=64, shuffle=True, drop_last=True)


def get_mean_std(loader):
    # VAR - variance of a variable
    # E - expected value of a variable
    # VAR[X] = E[X^2] - (E[X])^2
    #note **2 == ^2
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0  #Mnist has 1 channel, CIFAR10 has 3, works in both cases

    for data, _ in loader:  # _ is for targets, don't need
        #print(data.shape)
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1

    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches - mean**2)**0.5

    return mean, std

print("MNIST:")
mean, std = get_mean_std(mnist_loader)
print(mean)
print(std)


print("CIFAR10:")
mean, std = get_mean_std(cifar10_loader)
print(mean)
print(std)