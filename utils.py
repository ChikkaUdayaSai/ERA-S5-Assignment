import torch
from torchvision import datasets, transforms

def is_cuda_available() -> bool:
    """
    Check if CUDA is available on the system.

    Returns:
        bool: True if CUDA is available, False otherwise.
    """
    return torch.cuda.is_available()

def get_device() -> torch.device:
    """
    Get the device (CPU or GPU) to be used for training.

    Returns:
        torch.device: The device to be used for training.
    """
    if is_cuda_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


# Train data transformations
train_transforms = transforms.Compose([
    # Randomly apply center crop of size 22 to the image with probability 0.1
    transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
    # Resize the image to size 28x28
    transforms.Resize((28, 28)),
    # Randomly rotate the image by an angle between -15 and 15 degrees
    transforms.RandomRotation((-15., 15.), fill=0),
    # Convert the image to a tensor
    transforms.ToTensor(),
    # Normalize the image with mean and standard deviation of MNIST dataset
    transforms.Normalize((0.1307,), (0.3081,))
])

# Test data transformations
test_transforms = transforms.Compose([
    # Convert the image to a tensor
    transforms.ToTensor(),
    # Normalize the image with mean and standard deviation of MNIST dataset
    transforms.Normalize((0.1307,), (0.3081,))
    ])

# Load the MNIST train and test datasets and apply the respective transformations
train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)

# Set the batch size for the data loaders
batch_size = 512

# Set the data loader arguments
kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}

# Create the test and train data loaders
test_loader = torch.utils.data.DataLoader(test_data, **kwargs)
train_loader = torch.utils.data.DataLoader(train_data, **kwargs)