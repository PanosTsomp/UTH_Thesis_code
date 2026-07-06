from flask import Flask, request, render_template
import os
import nibabel as nib
import numpy as np
import cv2
import base64
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from werkzeug.utils import secure_filename

# --------------------
# FLASK SETUP
# --------------------
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --------------------
# CUSTOM METRICS (same as in your UNet code)
# --------------------
def dice_coef(y_true, y_pred, smooth=1.0):
    class_num = 4
    total_loss = 0
    for i in range(class_num):
        y_true_f = K.flatten(y_true[:, :, :, i])
        y_pred_f = K.flatten(y_pred[:, :, :, i])
        intersection = K.sum(y_true_f * y_pred_f)
        loss = ((2. * intersection + smooth) / 
                (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
        if i == 0:
            total_loss = loss
        else:
            total_loss += loss
    total_loss = total_loss / class_num
    return total_loss

def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:, :, :, 1] * y_pred[:, :, :, 1]))
    return (2. * intersection) / (
        K.sum(K.square(y_true[:, :, :, 1])) +
        K.sum(K.square(y_pred[:, :, :, 1])) + epsilon
    )

def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:, :, :, 2] * y_pred[:, :, :, 2]))
    return (2. * intersection) / (
        K.sum(K.square(y_true[:, :, :, 2])) +
        K.sum(K.square(y_pred[:, :, :, 2])) + epsilon
    )

def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:, :, :, 3] * y_pred[:, :, :, 3]))
    return (2. * intersection) / (
        K.sum(K.square(y_true[:, :, :, 3])) +
        K.sum(K.square(y_pred[:, :, :, 3])) + epsilon
    )

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(
        K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1))
    )
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

# --------------------
# LOAD MODEL
# --------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'recompiled_model.keras')
model = load_model(
    MODEL_PATH,
    custom_objects={
        "dice_coef": dice_coef,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "dice_coef_necrotic": dice_coef_necrotic,
        "dice_coef_edema": dice_coef_edema,
        "dice_coef_enhancing": dice_coef_enhancing
    },
    compile=False
)

# --------------------
# ALLOWED EXTENSIONS
# --------------------
ALLOWED_EXTENSIONS = {'nii', 'nii.gz'}

def allowed_file(filename):
    return ('.' in filename and 
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)

# --------------------
# ALTERNATIVE METRICS
# --------------------
def calculate_prediction_confidence(predictions):
    """
    Calculate the mean confidence of the model's predictions.
    :param predictions: Model's output (softmax probabilities), shape: (batch_size, height, width, num_classes)
    :return: Mean confidence across all pixels
    """
    max_probabilities = np.max(predictions, axis=-1)  # Get the maximum probability for each pixel
    mean_confidence = np.mean(max_probabilities)  # Calculate mean confidence
    return mean_confidence

def calculate_prediction_entropy(predictions):
    """
    Calculate the mean entropy of the model's predictions.
    :param predictions: Model's output (softmax probabilities), shape: (batch_size, height, width, num_classes)
    :return: Mean entropy across all pixels
    """
    epsilon = 1e-10  # Small value to avoid log(0)
    entropy = -np.sum(predictions * np.log(predictions + epsilon), axis=-1)  # Calculate entropy
    mean_entropy = np.mean(entropy)  # Calculate mean entropy
    return mean_entropy

def calculate_class_distribution(predictions):
    """
    Calculate the distribution of predicted classes.
    :param predictions: Predicted masks, shape: (height, width)
    :return: Dictionary of class frequencies
    """
    unique_classes, counts = np.unique(predictions, return_counts=True)
    class_distribution = dict(zip(unique_classes, counts / np.sum(counts)))
    return class_distribution

def calculate_boundary_smoothness(predicted_mask):
    """
    Calculate the smoothness of the predicted mask boundaries.
    :param predicted_mask: Predicted mask, shape: (height, width)
    :return: Total variation (lower is better)
    """
    gradient_x = np.abs(np.gradient(predicted_mask, axis=0))
    gradient_y = np.abs(np.gradient(predicted_mask, axis=1))
    total_variation = np.sum(gradient_x) + np.sum(gradient_y)
    return total_variation

# --------------------
# ROUTES
# --------------------
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Process the uploaded MRI file
            result = analyze_image(filepath)
            # Clean up the uploaded file after processing
            os.remove(filepath)
            return result
    return render_template('upload.html')

def analyze_image(filepath):
    mri_image = nib.load(filepath).get_fdata()  # Shape: (240, 240, 155)
    num_slices = mri_image.shape[2]
    image_slices = []
    all_predictions = []
    slice_metrics = []  # Store metrics for each slice
    
    for slice_idx in range(num_slices):
        mri_slice = cv2.resize(mri_image[:, :, slice_idx], (128, 128))
        mri_slice_resized = np.expand_dims(mri_slice, axis=(0, -1))
        mri_slice_resized = np.concatenate([mri_slice_resized, mri_slice_resized], axis=-1)
        prediction = model.predict(mri_slice_resized)
        predicted_mask = np.argmax(prediction, axis=-1)[0]
        all_predictions.append(predicted_mask)
        
        # Resize predicted mask back to original size
        predicted_mask_resized = cv2.resize(predicted_mask, (240, 240), interpolation=cv2.INTER_NEAREST)
        
        # Normalize the MRI slice for visualization
        max_value = np.max(mri_image[:, :, slice_idx])
        if max_value == 0:
            normalized_slice = np.zeros_like(mri_image[:, :, slice_idx], dtype=np.uint8)
        else:
            normalized_slice = (mri_image[:, :, slice_idx] / max_value * 255).astype(np.uint8)
        
        # Overlay the predicted mask on the MRI slice
        combined_image = cv2.addWeighted(
            cv2.cvtColor(normalized_slice, cv2.COLOR_GRAY2BGR),
            0.6,
            cv2.applyColorMap((predicted_mask_resized * 85).astype(np.uint8), cv2.COLORMAP_JET),
            0.4,
            0
        )
        
        # Encode the combined image as a base64 string
        _, buffer = cv2.imencode('.png', combined_image)
        image_slices.append(base64.b64encode(buffer).decode('utf-8'))
        
        # Calculate slice-specific metrics
        slice_confidence = calculate_prediction_confidence(prediction)
        slice_entropy = calculate_prediction_entropy(prediction)
        slice_class_distribution = calculate_class_distribution(predicted_mask)
        slice_boundary_smoothness = calculate_boundary_smoothness(predicted_mask)
        
        # Convert class distribution to a readable format
        readable_class_distribution = {
            f"Class {int(cls)}": f"{freq * 100:.2f}%"
            for cls, freq in slice_class_distribution.items()
        }
        
        # Store slice metrics
        slice_metrics.append({
            "confidence": f"{slice_confidence:.4f}",
            "entropy": f"{slice_entropy:.4f}",
            "class_distribution": readable_class_distribution,
            "boundary_smoothness": f"{slice_boundary_smoothness:.4f}"
        })
    
    # Render the result template with the image slices and metrics
    return render_template(
        'result.html',
        image_slices=image_slices,
        num_slices=num_slices,
        slice_metrics=slice_metrics,  # Pass slice-specific metrics to the template
        global_metrics={
            "mean_confidence": f"{calculate_prediction_confidence(np.array(all_predictions)):.4f}",
            "mean_entropy": f"{calculate_prediction_entropy(np.array(all_predictions)):.4f}",
            "class_distribution": {
                f"Class {int(cls)}": f"{freq * 100:.2f}%"
                for cls, freq in calculate_class_distribution(np.concatenate(all_predictions)).items()
            },
            "boundary_smoothness": f"{calculate_boundary_smoothness(np.concatenate(all_predictions)):.4f}"
        }
    )

if __name__ == '__main__':
    app.run(debug=True)