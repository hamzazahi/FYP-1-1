import os
import random

# Directory where your images are stored
image_dir = 'C:\\Users\\tasaw\\Desktop\\Yolov3-Custom-Training\\images'

# Getting all file names in the image directory
all_files = os.listdir(image_dir)
all_images = [file for file in all_files if file.endswith('.jpeg') or file.endswith('.png')]

print(f"Total images found: {len(all_images)}")  # Debugging line

# Shuffle the images to randomize
random.shuffle(all_images)

# Splitting data: 80% for training, 20% for validation
split = int(0.8 * len(all_images))
train_images = all_images[:split]
valid_images = all_images[split:]

print(f"Training images: {len(train_images)}")  # Debugging line
print(f"Validation images: {len(valid_images)}")  # Debugging line

# Function to write paths to file
def write_paths(file_path, image_names):
    with open(file_path, 'w') as file:
        for image in image_names:
            full_path = os.path.join(image_dir, image)
            file.write(full_path + '\n')

# Writing to train.txt and valid.txt
write_paths('C:\\Users\\tasaw\\Desktop\\Yolov3-Custom-Training\\train.txt', train_images)
write_paths('C:\\Users\\tasaw\\Desktop\\Yolov3-Custom-Training\\valid.txt', valid_images)

print("train.txt and valid.txt have been created.")
