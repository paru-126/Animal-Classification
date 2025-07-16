import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Check class distribution
dataset_path = 'dataset'
classes = os.listdir(dataset_path)

print("Number of images per class:")
for animal in classes:
    num_images = len(os.listdir(os.path.join(dataset_path, animal)))
    print(f"{animal}: {num_images} images")

# Visualize sample images
plt.figure(figsize=(15, 10))
for i, animal in enumerate(classes[:5]):  # Show first 5 classes
    img_path = os.path.join(dataset_path, animal, os.listdir(os.path.join(dataset_path, animal))[0])
    img = mpimg.imread(img_path)
    
    plt.subplot(1, 5, i+1)
    plt.imshow(img)
    plt.title(animal)
    plt.axis('off')

plt.tight_layout()
plt.show()
