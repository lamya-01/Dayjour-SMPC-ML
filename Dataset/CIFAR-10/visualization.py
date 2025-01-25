import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

# Load CIFAR-10
transform = transforms.Compose([transforms.ToTensor()])
cifar10_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

# Plot a few sample images
def plot_cifar10_samples(dataset, num_images=10):
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        img, label = dataset[i]
        img = img.permute(1, 2, 0)  # Rearrange dimensions for display (HWC)
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Label: {label}")
    plt.show()

plot_cifar10_samples(cifar10_data)

#statics
# Compute statistics for the CIFAR-10 dataset
data = [np.array(cifar10_data[i][0]) for i in range(len(cifar10_data))]  # Get all images as numpy arrays
data = np.stack(data, axis=0)  # Convert to a single numpy array

mean = data.mean(axis=(0, 2, 3))  # Mean for each channel (RGB)
std = data.std(axis=(0, 2, 3))  # Std deviation for each channel (RGB)

print("CIFAR-10 Summary Statistics:")
print(f"Mean (R, G, B): {mean}")
print(f"Std (R, G, B): {std}")

#Feature Correlation

# Compute average pixel intensities for each image
avg_pixels = data.mean(axis=(2, 3))  # Average over H and W for each channel

# Compute correlation matrix
correlation_matrix = np.corrcoef(avg_pixels.T)

import seaborn as sns
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=["R", "G", "B"], yticklabels=["R", "G", "B"])
plt.title("CIFAR-10 Channel Correlation")
plt.show()

#Class Imbalance
from collections import Counter

# Get label counts
labels = [cifar10_data[i][1] for i in range(len(cifar10_data))]
label_counts = Counter(labels)

# Plot class distribution
plt.bar(label_counts.keys(), label_counts.values())
plt.xticks(range(10), [f"Class {i}" for i in range(10)])
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("CIFAR-10 Class Distribution")
plt.show()

