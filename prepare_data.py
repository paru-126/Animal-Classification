from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
import shutil
import numpy as np
from collections import defaultdict

# Problem classes needing special augmentation
PROBLEM_CLASSES = ['Bear', 'Elephant', 'Deer', 'Kangaroo']

def create_dataset_structure(classes):
    base_dir = 'organized_data'
    os.makedirs(base_dir, exist_ok=True)
    
    print("Creating directory structure...")
    for subset in ['train', 'validation', 'test']:
        subset_dir = os.path.join(base_dir, subset)
        os.makedirs(subset_dir, exist_ok=True)
        for animal in classes:
            os.makedirs(os.path.join(subset_dir, animal), exist_ok=True)
    print("Directory structure created.")

def split_data(classes):
    class_counts = defaultdict(int)
    print("\nSplitting data with balanced distribution...")
    for animal in classes:
        src_dir = os.path.join('dataset', animal)
        images = os.listdir(src_dir)
        class_counts[animal] = len(images)
        
        train, temp = train_test_split(
            images, 
            test_size=0.3, 
            random_state=42,
            stratify=[animal]*len(images))
        val, test = train_test_split(
            temp, 
            test_size=0.5, 
            random_state=42,
            stratify=[animal]*len(temp))
        
        print(f"Processing {animal} ({len(images)} images)...")
        for img in train:
            shutil.copy(os.path.join(src_dir, img), os.path.join('organized_data', 'train', animal, img))
        for img in val:
            shutil.copy(os.path.join(src_dir, img), os.path.join('organized_data', 'validation', animal, img))
        for img in test:
            shutil.copy(os.path.join(src_dir, img), os.path.join('organized_data', 'test', animal, img))
    
    print("\nClass distribution in splits:")
    for animal, count in class_counts.items():
        print(f"{animal}: Train={int(count*0.7)}, Val={int(count*0.15)}, Test={int(count*0.15)}")

def create_generators():
    # Stronger augmentation for all classes (helps problem classes more)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        brightness_range=[0.7, 1.3],
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    
    # Standard generators for validation and test
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        'organized_data/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    
    validation_generator = test_datagen.flow_from_directory(
        'organized_data/validation',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    
    test_generator = test_datagen.flow_from_directory(
        'organized_data/test',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_generator)
    
    # Boost weights for problem classes
    for cls in PROBLEM_CLASSES:
        if cls in train_generator.class_indices:
            class_idx = train_generator.class_indices[cls]
            class_weights[class_idx] *= 1.5
    
    print("\nFinal class weights with problem class boosting:")
    for class_name, idx in train_generator.class_indices.items():
        print(f"{class_name}: {class_weights[idx]:.2f}")
    
    return train_generator, validation_generator, test_generator, class_weights

def calculate_class_weights(generator):
    class_counts = np.bincount(generator.classes)
    total_samples = generator.samples
    num_classes = len(generator.class_indices)
    
    class_weights = {
        i: total_samples / (num_classes * count) if count > 0 else 1.0
        for i, count in enumerate(class_counts)
    }
    return class_weights

if __name__ == '__main__':
    classes = sorted(os.listdir('dataset'))
    print(f"Found {len(classes)} animal classes: {', '.join(classes)}")
    create_dataset_structure(classes)
    split_data(classes)
    train_gen, val_gen, test_gen, class_weights = create_generators()
    np.save('class_weights.npy', class_weights)
    print("\nPreparation complete. Class weights saved to class_weights.npy")