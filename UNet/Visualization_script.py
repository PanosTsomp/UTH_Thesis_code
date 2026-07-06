# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import nibabel as nib
import cv2

# Define evaluation metrics
def dice_coef(y_true, y_pred, smooth=1.0):
    intersection = K.sum(K.abs(y_true * y_pred))
    return (2. * intersection + smooth) / (K.sum(K.square(y_true)) + K.sum(K.square(y_pred)) + smooth)

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

# Define color map for visualization
cmap = mcolors.ListedColormap(['black', 'yellow', 'green', 'red'])
bounds = [0, 1, 2, 3, 4]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Dataset path (adjust if necessary)
TRAIN_DATASET_PATH = r'C:\Users\Christos Tsoutsas\Desktop\University\BraTS2020\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData/'

# Load sample data for testing
sample_case_id = 'BraTS20_Training_001'
flair_path = os.path.join(TRAIN_DATASET_PATH, sample_case_id, f'{sample_case_id}_flair.nii')
t1ce_path = os.path.join(TRAIN_DATASET_PATH, sample_case_id, f'{sample_case_id}_t1ce.nii')
seg_path = os.path.join(TRAIN_DATASET_PATH, sample_case_id, f'{sample_case_id}_seg.nii')

# Load the images using nibabel
flair_image = nib.load(flair_path).get_fdata()
t1ce_image = nib.load(t1ce_path).get_fdata()
seg_image = nib.load(seg_path).get_fdata()

# Preprocess input image
IMG_SIZE = 128
slice_idx = 50  # Select slice index for visualization
input_image = np.stack([cv2.resize(flair_image[:, :, slice_idx], (IMG_SIZE, IMG_SIZE)),
                        cv2.resize(t1ce_image[:, :, slice_idx], (IMG_SIZE, IMG_SIZE))], axis=-1)
input_image = np.expand_dims(input_image / np.max(input_image), axis=0)  # Normalize and expand dimensions

# Load the trained model
model = load_model(
    r'C:\Users\Christos Tsoutsas\Desktop\University\UNet\model_x1_1.keras',  # Update with your actual model path if different
    custom_objects={
        'dice_coef': dice_coef,
        'precision': precision,
        'sensitivity': sensitivity,
        'specificity': specificity
    }
)

# Predict the segmentation mask
predicted_mask = model.predict(input_image)
predicted_mask = np.argmax(predicted_mask[0], axis=-1)  # Convert one-hot to class labels

# Visualize the input image, ground truth, and predicted mask
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

ax[0].imshow(flair_image[:, :, slice_idx], cmap='gray')
ax[0].set_title("Input Image (FLAIR)")

ax[1].imshow(seg_image[:, :, slice_idx], cmap=cmap, norm=norm)
ax[1].set_title("Ground Truth Mask")

ax[2].imshow(predicted_mask, cmap=cmap, norm=norm)
ax[2].set_title("Predicted Mask")

plt.colorbar(mcolors.ColorbarBase(ax=ax[2], cmap=cmap, norm=norm, ticks=[0, 1, 2, 3]))
plt.show()
