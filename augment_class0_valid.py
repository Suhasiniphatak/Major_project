from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import os

# ✅ Update these paths to match your dataset
input_dir = r'C:\Users\ADMIN\Documents\Project updated\aptos-augmented-images\valid\retina\0'   # Folder with 1433 images for class 0
output_dir = r'C:\Users\ADMIN\Documents\Project updated\aptos-augmented-images\valid\retina\0'  # Same folder (overwrite mode)

target_count = 2000
current_images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
current_count = len(current_images)

datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

print(f"Current images in class 0: {current_count}")
augmented_count = 0
i = 0

while current_count + augmented_count < target_count:
    img_path = os.path.join(input_dir, current_images[i % len(current_images)])
    img = load_img(img_path)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    for batch in datagen.flow(x, batch_size=1):
        save_path = os.path.join(input_dir, f'aug_{i}_{augmented_count}.jpg')
        array_to_img(batch[0]).save(save_path)
        augmented_count += 1
        break  # generate only one image per loop

    i += 1

print(f"✅ Augmented images created: {augmented_count}")
