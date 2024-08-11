import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.utils import CustomObjectScope
from data import load_data, tf_dataset
from train import iou

def read_image(path):
    """Read and preprocess image."""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (256, 256))
    image = image / 255.0  # Normalize to [0, 1]
    return image

def read_mask(path):
    """Read and preprocess mask."""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (256, 256))
    mask = np.expand_dims(mask, axis=-1)  # Expand dimensions to (256, 256, 1)
    return mask

def mask_parse(mask):
    """Convert mask to 3-channel image for visualization."""
    mask = np.squeeze(mask)  # Remove single channel
    mask = np.stack([mask, mask, mask], axis=-1)  # Convert to 3 channels
    return mask

if __name__ == "__main__":
    model_path = r"D:\Python\melanoma-skin-cancer-image-segmentation-master\files\model.keras"
    data_path = r"D:\Python\melanoma-skin-cancer-image-segmentation-master\dataset"
    output_folder = r"results\satyajit"  # Output folder for single segmented images
    collage_output_folder = r"results\satyajit_collage"  # Output folder for collage images
    batch_size = 14

   # Load data
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(data_path)

    # Ensure validation data is not empty
    if len(valid_x) == 0 or len(valid_y) == 0:
        raise ValueError("Validation data is empty. Please check the load_data function.")

    # Load model
    with CustomObjectScope({'iou': iou}):
        model = tf.keras.models.load_model(model_path)

    # Predict and save results with collage images
    for i, (x_path, y_path) in tqdm(enumerate(zip(valid_x, valid_y)), total=len(valid_x)):  # Using validation data
        # Read images
        original_img = read_image(x_path)
        ground_truth_mask = mask_parse(read_mask(y_path))

        # Predict segmented mask
        segmented_mask = model.predict(np.expand_dims(original_img, axis=0))[0] > 0.5
        segmented_mask = segmented_mask.astype(np.uint8) * 255
        segmented_mask = mask_parse(segmented_mask)

        #Save single segmented image
        os.makedirs(output_folder, exist_ok=True)
        cv2.imwrite(os.path.join(output_folder, f'{i}_segmented.png'), segmented_mask)

        # Create collage image
        h, w, _ = original_img.shape
        white_line = np.ones((h, 10, 3)) * 255.0
        all_images = [original_img * 255.0, white_line, ground_truth_mask, white_line, segmented_mask]

        # Save collage image
        collage = np.concatenate(all_images, axis=1)
        os.makedirs(collage_output_folder, exist_ok=True)
        cv2.imwrite(os.path.join(collage_output_folder, f'{i}_collage.png'), collage)

    print("Segmented images and collage images saved successfully.")
