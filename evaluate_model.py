import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import models
import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable
from sklearn.metrics import (classification_report, 
                           confusion_matrix,
                           precision_recall_curve,
                           average_precision_score)
import json
from PIL import Image
from collections import Counter
import time

# Register custom loss function
@register_keras_serializable()
def weighted_loss(y_true, y_pred):
    """Default weighted loss function implementation"""
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

try:
    import seaborn as sns
    sns.set(style='whitegrid', palette='muted')
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("Seaborn not found, using matplotlib only")

class ModelEvaluator:
    def __init__(self):
        self.model = None
        self.interpreter = None
        self.is_tflite = False
        self.test_gen = None
        self.class_names = []
        self.y_true = []
        self.y_pred = []
        self.input_details = None
        self.output_details = None
        
    def load_model(self, model_path='animal_classifier.keras'):
        """Load trained model (Keras or TFLite) with error handling"""
        try:
            if model_path.endswith('.tflite'):
                self.interpreter = tf.lite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                self.is_tflite = True
                print(f"Successfully loaded TFLite model from {model_path}")
            else:
                custom_objects = {'weighted_loss': weighted_loss}
                self.model = load_model(model_path, custom_objects=custom_objects)
                self.is_tflite = False
                print(f"Successfully loaded Keras model from {model_path}")
                print("Available layers:", [layer.name for layer in self.model.layers])
            
            if os.path.exists('class_indices.json'):
                with open('class_indices.json') as f:
                    class_indices = json.load(f)
                self.class_names = list(class_indices.keys())
                
        except Exception as e:
            raise ValueError(f"Error loading model: {str(e)}")
    
    def predict(self, images):
        """Unified prediction method for both Keras and TFLite"""
        if self.is_tflite:
            input_shape = self.input_details[0]['shape']
            if images.shape[1:3] != tuple(input_shape[1:3]):
                images = tf.image.resize(images, (input_shape[1], input_shape[2]))
            self.interpreter.set_tensor(self.input_details[0]['index'], images)
            self.interpreter.invoke()
            return self.interpreter.get_tensor(self.output_details[0]['index'])
        else:
            return self.model.predict(images, verbose=0)
    
    def load_test_data(self, test_dir='organized_data/test', img_size=(224, 224), batch_size=32):
        """Prepare test data generator"""
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        self.test_gen = test_datagen.flow_from_directory(
            test_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False)
        
        if not self.class_names:
            self.class_names = list(self.test_gen.class_indices.keys())
            
        with open('class_indices.json', 'w') as f:
            json.dump(self.test_gen.class_indices, f)
    
    def benchmark_performance(self, num_runs=100):
        """Measure inference speed"""
        if not self.test_gen:
            raise ValueError("Test data not loaded")
            
        test_image = next(self.test_gen)[0][0:1]
        
        # Warmup
        _ = self.predict(test_image)
        
        start = time.time()
        for _ in range(num_runs):
            _ = self.predict(test_image)
        avg_time = (time.time() - start)/num_runs
        
        print(f"\nAverage inference time: {avg_time*1000:.2f}ms")
        return avg_time
    
    def _find_conv_layer(self):
        """Find suitable convolutional layer for Grad-CAM"""
        for layer in reversed(self.model.layers):
            if 'conv' in layer.name.lower() and len(layer.output_shape) == 4:
                return layer.name
        return None
    
    def generate_gradcam(self, img_path):
        """Generate Grad-CAM heatmap for visual explanations"""
        if self.is_tflite:
            print("Grad-CAM not supported for TFLite models")
            return None
            
        layer_name = self._find_conv_layer()
        if not layer_name:
            print("Could not find suitable convolutional layer for Grad-CAM")
            return None
            
        print(f"Using layer '{layer_name}' for Grad-CAM")
        
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        grad_model = models.Model(
            [self.model.inputs],
            [self.model.get_layer(layer_name).output, self.model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, np.argmax(predictions[0])]
        
        grads = tape.gradient(loss, conv_outputs)[0]
        weights = tf.reduce_mean(grads, axis=(0, 1))
        cam = tf.reduce_sum(conv_outputs * weights, axis=-1).numpy()
        
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam = np.uint8(255 * cam)
        cam = np.array(Image.fromarray(cam).resize((224,224), Image.Resampling.LANCZOS))
        
        plt.figure(figsize=(10, 10))
        plt.imshow(img_array[0])
        plt.imshow(cam, cmap='jet', alpha=0.5)
        plt.axis('off')
        plt.title(f'Grad-CAM Visualization (Layer: {layer_name})', pad=20)
        plt.savefig('gradcam.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        return cam
    
    def evaluate(self):
        """Run comprehensive evaluation"""
        if not (self.model or self.interpreter):
            raise ValueError("Model not loaded")
        if not self.test_gen:
            raise ValueError("Test data not loaded")
            
        print("\n=== Basic Evaluation ===")
        if not self.is_tflite:
            evaluation_results = self.model.evaluate(self.test_gen, return_dict=True, verbose=0)
            print("Test Metrics:")
            for metric, value in evaluation_results.items():
                print(f"{metric}: {value:.4f}")
            accuracy = evaluation_results.get('accuracy', 0)
            loss = evaluation_results['loss']
        else:
            correct = 0
            total = 0
            losses = []
            
            self.test_gen.reset()
            for _ in range(len(self.test_gen)):
                x, y = next(self.test_gen)
                preds = self.predict(x)
                correct += np.sum(np.argmax(preds, axis=1) == np.argmax(y, axis=1))
                total += len(y)
                losses.append(tf.keras.losses.categorical_crossentropy(y, preds))
            
            accuracy = correct / total
            loss = np.mean(losses)
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Test Loss: {loss:.4f}")
        
        self.test_gen.reset()
        self.y_true = self.test_gen.classes
        self.y_pred = self.predict(self.test_gen)
        y_pred_classes = np.argmax(self.y_pred, axis=1)
        
        print("\n=== Classification Report ===")
        print(classification_report(
            self.y_true, 
            y_pred_classes, 
            target_names=self.class_names,
            digits=3))
        
        self._plot_confusion_matrix(y_pred_classes)
        self._plot_precision_recall()
        self._plot_sample_predictions()
        errors = self._analyze_errors(y_pred_classes)
        self.benchmark_performance()
        
        if len(errors) > 0 and not self.is_tflite:
            self.generate_gradcam(self.test_gen.filepaths[errors[0]])
        
        self._save_report(accuracy, loss)
    
    def _plot_confusion_matrix(self, y_pred_classes):
        cm = confusion_matrix(self.y_true, y_pred_classes)
        plt.figure(figsize=(15, 12))
        
        if SEABORN_AVAILABLE:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.class_names,
                       yticklabels=self.class_names,
                       annot_kws={"size": 8})
        else:
            plt.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.colorbar()
        
        plt.title('Confusion Matrix', pad=20)
        plt.xlabel('Predicted', labelpad=10)
        plt.ylabel('Actual', labelpad=10)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_precision_recall(self):
        plt.figure(figsize=(12, 8))
        
        for i, class_name in enumerate(self.class_names):
            precision, recall, _ = precision_recall_curve(
                (self.y_true == i).astype(int),
                self.y_pred[:, i])
            ap = average_precision_score(
                (self.y_true == i).astype(int), 
                self.y_pred[:, i])
            
            plt.plot(recall, precision, 
                    label=f'{class_name} (AP={ap:.2f})',
                    linewidth=2)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall Curves by Class', pad=20)
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_sample_predictions(self, num_samples=9):
        self.test_gen.reset()
        images, labels = next(self.test_gen)
        predictions = self.predict(images[:num_samples])
        
        plt.figure(figsize=(15, 15))
        for i in range(min(num_samples, len(images))):
            plt.subplot(3, 3, i+1)
            plt.imshow(images[i])
            
            true_idx = np.argmax(labels[i])
            pred_idx = np.argmax(predictions[i])
            confidence = np.max(predictions[i])
            
            color = 'green' if true_idx == pred_idx else 'red'
            plt.title(f"True: {self.class_names[true_idx]}\nPred: {self.class_names[pred_idx]}\nConf: {confidence:.1%}", 
                     color=color)
            plt.axis('off')
        
        plt.suptitle('Sample Predictions with Confidence Scores', y=1.02)
        plt.tight_layout()
        plt.savefig('sample_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _analyze_errors(self, y_pred_classes):
        errors = np.where(self.y_true != y_pred_classes)[0]
        if len(errors) == 0:
            print("\nPerfect classification on test set!")
            return errors
            
        print(f"\n=== Error Analysis ({len(errors)} errors) ===")
        
        error_pairs = [(self.class_names[self.y_true[i]], 
                       self.class_names[y_pred_classes[i]]) 
                      for i in errors]
        
        common_errors = Counter(error_pairs).most_common(5)
        
        print("\nTop 5 Misclassifications:")
        for (true, pred), count in common_errors:
            print(f"{true.ljust(10)} → {pred.ljust(10)}: {count} cases")
        
        os.makedirs('problematic_images', exist_ok=True)
        for i in errors[:5]:
            img_path = self.test_gen.filepaths[i]
            img = Image.open(img_path)
            save_name = f"error_{i}_{self.class_names[self.y_true[i]]}_as_{self.class_names[y_pred_classes[i]]}.png"
            img.save(os.path.join('problematic_images', save_name))
        
        print("\nSaved problematic images to 'problematic_images' directory")
        return errors
    
    def _save_report(self, accuracy, loss):
        report = {
            "test_accuracy": float(accuracy),
            "test_loss": float(loss),
            "class_names": self.class_names,
            "confusion_matrix": confusion_matrix(self.y_true, np.argmax(self.y_pred, axis=1)).tolist(),
            "classification_report": classification_report(
                self.y_true, 
                np.argmax(self.y_pred, axis=1),
                target_names=self.class_names,
                output_dict=True),
            "problematic_classes": self._identify_problem_classes()
        }
        
        with open('evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("\nSaved complete evaluation report to 'evaluation_report.json'")
    
    def _identify_problem_classes(self):
        report = classification_report(
            self.y_true,
            np.argmax(self.y_pred, axis=1),
            target_names=self.class_names,
            output_dict=True)
        
        problem_classes = []
        for class_name in self.class_names:
            if report[class_name]['f1-score'] < 0.7:
                problem_classes.append({
                    'class': class_name,
                    'precision': report[class_name]['precision'],
                    'recall': report[class_name]['recall'],
                    'f1': report[class_name]['f1-score'],
                    'support': report[class_name]['support']
                })
        
        return problem_classes

if __name__ == '__main__':
    evaluator = ModelEvaluator()
    try:
        evaluator.load_model()
        evaluator.load_test_data()
        evaluator.evaluate()
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        raise