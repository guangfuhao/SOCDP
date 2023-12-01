import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
from configs import *
import random


def display_images(data_loader, title='16 images from train data'):
    images, labels = next(iter(data_loader))
    if len(images) > 16:
        images = images[:16]
        labels = labels[:16]

    plt.figure(figsize=(8, 8))
    for idx, (img, lbl) in enumerate(zip(images, labels)):
        ax = plt.subplot(4, 4, idx + 1)
        img_np = img.numpy().transpose(1, 2, 0)
        ax.imshow(img_np, cmap='gray')
        ax.axis('off')
        ax.set_title(str(lbl.numpy()))

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def generate_location_list(num_loc=10, show=False):
    random.seed(12)
    numbers = list(range(1, 33))
    loc_list = [[] for _ in range(num_loc)]

    for i in range(1, num_loc + 1):
        selected_numbers = sorted(random.sample(numbers, 10))
        for num in selected_numbers:
            if num <= 9:
                loc_list[i - 1].append((0, (num - 1) * 3))
            elif num <= 18:
                loc_list[i - 1].append((25, (num - 10) * 3))
            elif num <= 25:
                loc_list[i - 1].append(((num - 18) * 3, 0))
            else:
                loc_list[i - 1].append(((num - 25) * 3, 25))

    if show:
        print(loc_list)
    return loc_list


def load_processed_dataset(data_dir='./data/mnist/', batch_size=50, display=False, process=False, class_num=100):
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        download_mnist = True
    else:
        download_mnist = False

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.1307, std=0.3081)])

    train_data_raw = datasets.MNIST(root=data_dir, transform=transform, train=True, download=download_mnist)
    test_data_raw = datasets.MNIST(root=data_dir, transform=transform, train=False)

    if process:
        loc = LOC_LIST
        for dataset in [train_data_raw, test_data_raw]:
            indices = list(range(len(dataset)))
            np.random.shuffle(indices)
            splits = np.array_split(indices, int(class_num / 10))
            for idx, split in enumerate(splits):
                dataset.targets[split] += 10 * idx
                for loc_idx in loc[idx]:
                    x_start, y_start = loc_idx
                    dataset.data[split, x_start:x_start + 3, y_start:y_start + 3] = 255

        train_data = DataLoader(dataset=train_data_raw, batch_size=batch_size, shuffle=True)
        test_data = DataLoader(dataset=test_data_raw, batch_size=batch_size, shuffle=False)

    else:
        train_data = DataLoader(dataset=train_data_raw, batch_size=batch_size, shuffle=True)
        test_data = DataLoader(dataset=test_data_raw, batch_size=batch_size, shuffle=False)

    if display:
        display_images(train_data)

    return train_data, test_data


def dot_without_number_display_images(disp=False):
    images = []
    loc_list = LOC_LIST
    num_images = len(loc_list)
    image_size = (28, 28)
    for i in range(num_images):
        # Create a black 28x28 image
        image = np.zeros(image_size, dtype=np.uint8)
        # Get the locations for this image (cycling through LOC_LIST)
        loc_idx = i % len(loc_list)
        for loc in loc_list[loc_idx]:
            x_start, y_start = loc
            # Set the 3x3 region to white
            image[x_start:x_start + 3, y_start:y_start + 3] = 255
        # Add the modified image to the list
        images.append(image)
    if disp:
        # Now display the images
        fig, axes = plt.subplots(3, 3, figsize=(6, 6), facecolor='white')  # Set the background color to black
        # Set the spacing between the subplots
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        # Flatten the axes array for easy 1D indexing
        axes_flat = axes.flatten()
        for idx in range(9):
            # Display each image
            axes_flat[idx].imshow(images[idx], cmap='gray')
            axes_flat[idx].axis('off')  # Hide the axes
            axes_flat[idx].set_facecolor('black')  # Keep the subplot background color black as well
        return images, fig
    else:
        return images
