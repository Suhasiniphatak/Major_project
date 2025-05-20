import os

# Path to the train folder
train_dir = r"C:\\Users\\ADMIN\\Documents\\Project updated\\aptos-augmented-images\\test"

# List the folders in the train directory (i.e., 0, 1, 2, 3, 4)
class_names = ['0', '1', '2', '3', '4']

# Dictionary to store image counts for each class
image_counts = {}

# Loop through each class folder (0, 1, 2, 3, 4) and count images
for class_name in class_names:
    class_path = os.path.join(train_dir, class_name)
    
    if os.path.isdir(class_path):  # Check if it's a valid directory
        # Count the number of images (files with extensions .jpg, .jpeg, .png)
        image_count = len([f for f in os.listdir(class_path) if f.endswith(('jpg', 'jpeg', 'png'))])
        image_counts[class_name] = image_count

# Display the number of images in each class
for class_name, count in image_counts.items():
    print(f"Class {class_name}: {count} images")

# Check if all classes have the same number of images
if len(set(image_counts.values())) == 1:
    print("\nAll classes have the same number of images.")
else:
    print("\nClasses have different numbers of images.")
