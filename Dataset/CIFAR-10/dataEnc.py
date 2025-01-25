import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import tenseal as ts
import numpy as np
import pickle
import os

# Define transformations for CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # Mean and std for CIFAR-10
])

# Create a TenSEAL context
context = ts.context(
    ts.SCHEME_TYPE.CKKS,  # Encryption scheme
    poly_modulus_degree=8192,  # Polynomial modulus degree
    coeff_mod_bit_sizes=[40, 21, 21, 21, 21]  # Coefficient modulus bit sizes
)
context.global_scale = 2**21  # Global scale for CKKS
context.generate_galois_keys()  # Generate Galois keys for operations like rotations

# Load CIFAR-10 data
cifar10_data = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
cifar10_loader = DataLoader(cifar10_data, batch_size=8, shuffle=True)

# Save encrypted dataset to a file
def save_encrypted_dataset(file_path, encrypted_data):
    serialized_data = []
    for batch in encrypted_data:
        serialized_batch = []
        for encrypted_img, label in batch:
            serialized_batch.append((encrypted_img.serialize(), label))  # Serialize encrypted vector
        serialized_data.append(serialized_batch)
    
    with open(file_path, 'wb') as f:
        pickle.dump(serialized_data, f)
    print(f"Encrypted dataset saved to {file_path}")

# Load encrypted dataset from a file
def load_encrypted_dataset(file_path, context):
    with open(file_path, 'rb') as f:
        serialized_data = pickle.load(f)
    
    loaded_data = []
    for batch in serialized_data:
        loaded_batch = []
        for serialized_img, label in batch:
            encrypted_img = ts.ckks_vector_from(context, serialized_img)  # Deserialize encrypted vector
            loaded_batch.append((encrypted_img, label))
        loaded_data.append(loaded_batch)
    print(f"Encrypted dataset loaded from {file_path}")
    return loaded_data

# File path to store the encrypted dataset
encrypted_data_file = "encrypted_cifar10.pkl"

# Check if the encrypted data file already exists
if os.path.exists(encrypted_data_file):
    print("Loading encrypted dataset from file...")
    encrypted_data = load_encrypted_dataset(encrypted_data_file, context)
else:
    print("Encrypting dataset for the first time...")
    encrypted_data = []
    for images, labels in cifar10_loader:
        encrypted_batch = []
        for img in images:
            flattened_img = img.view(-1).numpy()  # Flatten image
            encrypted_img = ts.ckks_vector(context, flattened_img)  # Encrypt image
            encrypted_batch.append((encrypted_img, labels.numpy()))
        encrypted_data.append(encrypted_batch)
    
    # Save the encrypted dataset to a file
    save_encrypted_dataset(encrypted_data_file, encrypted_data)

# Example: Inspect the first encrypted image
print("Encrypted data sample:", encrypted_data[0][0][0])  # Access the first encrypted image
print("Label:", encrypted_data[0][0][1])  # Corresponding label
