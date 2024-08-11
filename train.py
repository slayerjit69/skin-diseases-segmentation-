import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.utils import CustomObjectScope
from data import load_data, tf_dataset

def iou(y_true, y_pred):
    y_true = tf.cast(y_true, tf.bool)
    y_pred = tf.cast(y_pred, tf.bool)
    intersection = tf.logical_and(y_true, y_pred)
    union = tf.logical_or(y_true, y_pred)
    iou_score = tf.reduce_sum(tf.cast(intersection, tf.float32)) / tf.reduce_sum(tf.cast(union, tf.float32))
    return iou_score

def read_image(path):
    """Read and preprocess image."""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (256, 256))
    image = image / 255.0  # Normalize to [0, 1]
    #print(f"Image shape: {image.shape}, min: {image.min()}, max: {image.max()}")
    return image

def read_mask(path):
    """Read and preprocess mask."""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (256, 256))
    mask = np.expand_dims(mask, axis=-1)  # Expand dimensions to (256, 256, 1)
    #print(f"Mask shape: {mask.shape}, unique values: {np.unique(mask)}")
    return mask

def mask_parse(mask):
    """Convert mask to 3-channel image for visualization."""
    mask = np.squeeze(mask)  # Remove single channel
    mask = np.stack([mask, mask, mask], axis=-1)  # Convert to 3 channels
    return mask

def visualize_images(images, masks, predictions=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    for i in range(len(images)):
        plt.subplot(3, len(images), i + 1)
        plt.imshow(images[i])
        plt.title("Image")
        plt.axis('off')

        plt.subplot(3, len(images), i + 1 + len(images))
        plt.imshow(masks[i].squeeze(), cmap='gray')
        plt.title("Mask")
        plt.axis('off')

        if predictions is not None:
            plt.subplot(3, len(images), i + 1 + 2 * len(images))
            plt.imshow(predictions[i].squeeze(), cmap='gray')
            plt.title("Prediction")
            plt.axis('off')
    plt.show()

if __name__ == "__main__":
    model_path = r"D:\Python\melanoma-skin-cancer-image-segmentation-master\files\model.keras"
    data_path = r"D:\Python\melanoma-skin-cancer-image-segmentation-master\dataset"
    batch_size = 14

    # Load data
    (train_x, train_y), (test_x, test_y), (valid_x, valid_y) = load_data(data_path)

    # Debug: Check lengths of data
    print(f"Length of test_x: {len(test_x)}")
    print(f"Length of test_y: {len(test_y)}")

    # Ensure test data is not empty
    if len(test_x) == 0 or len(test_y) == 0:
        raise ValueError("Test data is empty. Please check the load_data function.")

    # Create dataset
    test_dataset = tf_dataset(test_x, test_y, batch=batch_size)
    test_steps = len(test_x) // batch_size
    if len(test_x) % batch_size != 0:
        test_steps += 1

    # Debug: Print test_steps
    print(f"Test steps: {test_steps}")

    # Load model
    with CustomObjectScope({'iou': iou}):
        model = tf.keras.models.load_model(model_path)

    # Evaluate model
    model.evaluate(test_dataset, steps=test_steps)

    # Predict and save results
    for i, (x_path, y_path) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        x = read_image(x_path)
        y = read_mask(y_path)
        y_pred = model.predict(np.expand_dims(x, axis=0))[0]
        
        # Debugging: Print prediction stats
        #print(f"Prediction shape: {y_pred.shape}, min: {y_pred.min()}, max: {y_pred.max()}")

        y_pred = y_pred > 0.5  # Apply threshold
        y_pred = y_pred.astype(np.uint8) * 255  # Convert to 8-bit image

        if not np.any(y_pred):  # Check if prediction is completely black
            print(f"Warning: Prediction mask is completely black for image {i}")

        # Save the segmented image in the segmented_images folder
        os.makedirs('dataset/segmented_images', exist_ok=True)
        cv2.imwrite(os.path.join('dataset/segmented_images', f'{i}.png'), y_pred)

    print("Segmentation results saved successfully.")

    # Visualize a few samples
    samples = 4
    sample_images = [read_image(test_x[i]) for i in range(samples)]
    sample_masks = [read_mask(test_y[i]) for i in range(samples)]
    sample_preds = [model.predict(np.expand_dims(img, axis=0))[0] > 0.5 for img in sample_images]

    visualize_images(sample_images, sample_masks, sample_preds)
