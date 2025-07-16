from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from prepare_data import create_generators
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.saving import register_keras_serializable

# Problem classes to focus on
PROBLEM_CLASSES = ['Bear', 'Elephant', 'Deer', 'Kangaroo']

@register_keras_serializable()
def focal_loss(y_true, y_pred, class_weights, gamma=2.0):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    cross_entropy = -y_true * tf.math.log(y_pred)
    weights = tf.reduce_sum(class_weights * y_true, axis=1)
    p = tf.reduce_sum(y_pred * y_true, axis=1)
    focal_factor = tf.pow(1.0 - p, gamma)
    loss = focal_factor * cross_entropy
    loss = tf.reduce_sum(loss, axis=1)
    return tf.reduce_mean(weights * loss)

def create_focus_generators():
    train_gen, val_gen, _, class_weights = create_generators()
    
    # Convert weights to array
    weights_array = np.ones(len(class_weights))
    for idx, weight in class_weights.items():
        weights_array[idx] = weight * (3.0 if idx in [train_gen.class_indices[cls] for cls in PROBLEM_CLASSES] else 1.0)
    
    print("\nEnhanced class weights:")
    for cls, idx in train_gen.class_indices.items():
        marker = " (FOCUS)" if cls in PROBLEM_CLASSES else ""
        print(f"{cls}: {weights_array[idx]:.2f}{marker}")
    
    return train_gen, val_gen, weights_array

def train_focus_model():
    # Create fresh model if no existing one
    from build_model import build_model
    model = build_model()
    
    # Get data with enhanced focus
    train_gen, val_gen, class_weights = create_focus_generators()
    
    # Create loss function with current weights
    def current_focal_loss(y_true, y_pred):
        return focal_loss(y_true, y_pred, class_weights)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.00005),
        loss=current_focal_loss,
        metrics=['accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_recall', patience=8, restore_best_weights=True),
        ModelCheckpoint('focused_model.keras', 
                       save_best_only=True,
                       monitor='val_recall',
                       mode='max')
    ]
    
    # Train
    print("\nStarting focused training...")
    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // train_gen.batch_size,
        validation_data=val_gen,
        validation_steps=val_gen.samples // val_gen.batch_size,
        epochs=30,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\nFocused training complete. Model saved as 'focused_model.keras'")
    return history

if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    history = train_focus_model()