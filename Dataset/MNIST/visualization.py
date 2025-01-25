import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

# Load MNIST
mnist_data = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())

#Important Visualizations
# Plot a few sample images
def plot_mnist_samples(dataset, num_images=10):
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        img, label = dataset[i]
        img = img.squeeze(0)  # Remove channel dimension
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.title(f"Label: {label}")
    plt.show()

plot_mnist_samples(mnist_data)

#Summary Statistics
# Compute statistics for the MNIST dataset
data = [np.array(mnist_data[i][0]) for i in range(len(mnist_data))]  # Get all images as numpy arrays
data = np.stack(data, axis=0)  # Convert to a single numpy array

mean = data.mean()  # Mean of all pixel values
std = data.std()  # Std deviation of all pixel values

print("MNIST Summary Statistics:")
print(f"Mean: {mean}")
print(f"Std: {std}")

#Feature Correlation
# Compute average pixel intensity per image
avg_pixels = data.mean(axis=(1, 2))  # Average over H and W


# Plot histogram of average pixel intensities
plt.hist(avg_pixels, bins=30, color="blue", alpha=0.7)
plt.title("Histogram of Average Pixel Intensities (MNIST)")
plt.xlabel("Average Intensity")
plt.ylabel("Frequency")
plt.show()

# Get label counts
labels = [mnist_data[i][1] for i in range(len(mnist_data))]
label_counts = Counter(labels)

#Class Imbalance
# Plot class distribution
plt.bar(label_counts.keys(), label_counts.values(), color="orange")
plt.xticks(range(10), [str(i) for i in range(10)])
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("MNIST Class Distribution")
plt.show()
