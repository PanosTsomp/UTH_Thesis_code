# Standard libraries
import os
import shutil
import glob
import random

# Data manipulation and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageOps
import wandb
from wandb.integration.keras import WandbCallback


# Image processing
import cv2
from skimage.util import montage
from skimage.transform import rotate, resize

# Medical imaging
import nibabel as nib
import nilearn as nl
import nilearn.plotting as nlplt

# Machine learning and deep learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, concatenate, Layer, Dense, Dropout, LayerNormalization, MultiHeadAttention
from keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard, CSVLogger
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow.keras.layers as layers
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import concatenate

# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

# Define segmentation classes
SEGMENT_CLASSES = {
    0: 'NOT tumor',
    1: 'NECROTIC/CORE',  # or NON-ENHANCING tumor CORE
    2: 'EDEMA',
    3: 'ENHANCING'  # original 4 -> converted into 3 later
}

# Constants
VOLUME_SLICES = 100
VOLUME_START_AT = 22  # first slice of volume that we will include
TRAIN_DATASET_PATH = r'C:\Users\Christos Tsoutsas\Desktop\University\BraTS2020\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData/'
VALIDATION_DATASET_PATH = r'C:\Users\Christos Tsoutsas\Desktop\University\BraTS2020\BraTS2020_ValidationData'
IMG_SIZE = 128

# Load sample data
test_image_flair = nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_flair.nii').get_fdata()
test_image_t1 = nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t1.nii').get_fdata()
test_image_t1ce = nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t1ce.nii').get_fdata()
test_image_t2 = nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t2.nii').get_fdata()
test_mask = nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_seg.nii').get_fdata()

# Plot sample data
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 10))
slice_w = 25
ax1.imshow(test_image_flair[:, :, test_image_flair.shape[0] // 2 - slice_w], cmap='gray')
ax1.set_title('Image flair')
ax2.imshow(test_image_t1[:, :, test_image_t1.shape[0] // 2 - slice_w], cmap='gray')
ax2.set_title('Image t1')
ax3.imshow(test_image_t1ce[:, :, test_image_t1ce.shape[0] // 2 - slice_w], cmap='gray')
ax3.set_title('Image t1ce')
ax4.imshow(test_image_t2[:, :, test_image_t2.shape[0] // 2 - slice_w], cmap='gray')
ax4.set_title('Image t2')
ax5.imshow(test_mask[:, :, test_mask.shape[0] // 2 - slice_w])
ax5.set_title('Mask')
plt.show()

# Skip 50:-50 slices since there is not much to see
fig, ax1 = plt.subplots(1, 1, figsize=(15, 15))
ax1.imshow(rotate(montage(test_image_t1[50:-50, :, :]), 90, resize=True), cmap='gray')
plt.show()

# Skip 50:-50 slices since there is not much to see
fig, ax1 = plt.subplots(1, 1, figsize=(15, 15))
ax1.imshow(rotate(montage(test_mask[60:-60, :, :]), 90, resize=True), cmap='gray')
plt.show()


# Plot anatomical images
niimg = nl.image.load_img(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_flair.nii')
nimask = nl.image.load_img(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_seg.nii')

fig, axes = plt.subplots(nrows=4, figsize=(30, 40))
nlplt.plot_anat(niimg, title='BraTS20_Training_001_flair.nii plot_anat', axes=axes[0])
nlplt.plot_epi(niimg, title='BraTS20_Training_001_flair.nii plot_epi', axes=axes[1])
nlplt.plot_img(niimg, title='BraTS20_Training_001_flair.nii plot_img', axes=axes[2])
nlplt.plot_roi(nimask, title='BraTS20_Training_001_flair.nii with mask plot_roi', bg_img=niimg, axes=axes[3], cmap='Paired')
plt.show()

# Dice coefficient
def dice_coef(y_true, y_pred, smooth=1.0):
    class_num = 4
    total_loss = 0.0
    
    # Convert inputs to float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    for i in range(class_num):
        y_true_f = K.flatten(y_true[:, :, :, i])
        y_pred_f = K.flatten(y_pred[:, :, :, i])
        intersection = K.sum(y_true_f * y_pred_f)
        loss = ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
        total_loss += loss
    
    return total_loss / class_num

def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = K.sum(K.abs(y_true[:, :, :, 1] * y_pred[:, :, :, 1]))
    return (2. * intersection) / (K.sum(K.square(y_true[:, :, :, 1])) + K.sum(K.square(y_pred[:, :, :, 1])) + epsilon)

def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = K.sum(K.abs(y_true[:, :, :, 2] * y_pred[:, :, :, 2]))
    return (2. * intersection) / (K.sum(K.square(y_true[:, :, :, 2])) + K.sum(K.square(y_pred[:, :, :, 2])) + epsilon)

def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = K.sum(K.abs(y_true[:, :, :, 3] * y_pred[:, :, :, 3]))
    return (2. * intersection) / (K.sum(K.square(y_true[:, :, :, 3])) + K.sum(K.square(y_pred[:, :, :, 3])) + epsilon)

def precision(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def sensitivity(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

# Build ViT model

class PatchEmbedding(Layer):
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.projection = Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size)
        
    def call(self, x):
        patches = self.projection(x)
        shape = tf.shape(patches)
        batch_size, h, w, c = shape[0], shape[1], shape[2], shape[3]
        patches = tf.reshape(patches, [batch_size, h * w, c])
        return patches
    

class PositionalEmbedding(Layer):
    def __init__(self, seq_length, embed_dim):
        super(PositionalEmbedding, self).__init__()
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.position_embeddings = layers.Embedding(input_dim=seq_length, output_dim=embed_dim)

    def call(self, inputs):
        positions = tf.range(start=0, limit=self.seq_length, delta=1)
        position_embeddings = self.position_embeddings(positions)
        return position_embeddings

class ReshapeLayer(Layer):
    def __init__(self, target_shape):
        super(ReshapeLayer, self).__init__()
        self.target_shape = target_shape

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        reshaped = tf.reshape(inputs, [batch_size] + self.target_shape)
        return reshaped


class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.mlp = tf.keras.Sequential([
            Dense(mlp_dim, activation="gelu"),
            Dropout(dropout),
            Dense(embed_dim),
            Dropout(dropout)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(dropout)
        
    def call(self, inputs, training):
        x = self.layernorm1(inputs)
        attn_output = self.att(x, x)
        x = attn_output + inputs
        
        x2 = self.layernorm2(x)
        x2 = self.mlp(x2)
        return x + x2

def build_vit(input_shape=(128, 128, 2), patch_size=16, embed_dim=256, num_heads=8, 
              num_transformer_layers=4, mlp_dim=512, num_classes=4, dropout=0.1):
    inputs = Input(input_shape)
    input_size = input_shape[0]  # Should be 128
    
    # Patch Embedding
    patches = PatchEmbedding(patch_size, embed_dim)(inputs)
    
    # Positional Embedding
    seq_length = input_shape[0] // patch_size * input_shape[1] // patch_size
    positional_embedding_layer = PositionalEmbedding(seq_length=seq_length, embed_dim=embed_dim)
    position_embeddings = positional_embedding_layer(patches)
    x = patches + position_embeddings

    # Store skip connections
    skip_connections = []
    
    # Transformer Encoder with skip connections
    for i in range(num_transformer_layers):
        x = TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)(x, training=True)
        skip_connections.append(x)
    
    # Initial reshape to 2D
    x = ReshapeLayer([input_size // patch_size, input_size // patch_size, embed_dim])(x)
    current_size = input_size // patch_size  # Should be 16 for 128x128 input
    
    # Decoder with skip connections
    decoder_filters = [128, 64, 32, 16]
    
    for i, filters in enumerate(decoder_filters):
        # Calculate target size for this stage
        target_size = min(current_size * 2, input_size)
        
        # Upsample main path
        if current_size < input_size:
            x = UpSampling2D(size=(2, 2))(x)
            current_size *= 2
        
        x = Conv2D(filters, 3, padding='same', activation='relu')(x)
        
        if i < len(skip_connections):
            # Process skip connection
            skip = skip_connections[-(i+1)]
            skip = ReshapeLayer([input_size // patch_size, input_size // patch_size, embed_dim])(skip)
            
            # Upsample skip if needed
            if current_size > input_size // patch_size:
                skip_upsample_factor = current_size // (input_size // patch_size)
                skip = UpSampling2D(size=(skip_upsample_factor, skip_upsample_factor))(skip)
            
            # Process skip features
            skip = Conv2D(filters, 3, padding='same', activation='relu')(skip)
            
            # Concatenate
            x = concatenate([x, skip])
        
        # Additional processing
        x = Conv2D(filters, 3, padding='same', activation='relu')(x)
        
        print(f"Stage {i}: Current size = {current_size}, Shape = {x.shape}")
    
    # Final classification
    outputs = Conv2D(num_classes, 1, activation='softmax')(x)
    
    return Model(inputs=inputs, outputs=outputs)




# Define SGD optimizer with momentum
optimizer = SGD(learning_rate=0.001, momentum=0.9, nesterov=True)

# Define input layer and compile model
input_layer = Input((IMG_SIZE, IMG_SIZE, 2))
model = build_vit(input_shape=(IMG_SIZE, IMG_SIZE, 2))
model.compile(loss="categorical_crossentropy", optimizer=optimizer,
              metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=4), dice_coef, precision, sensitivity, specificity,
                       dice_coef_necrotic, dice_coef_edema, dice_coef_enhancing])


# Learning Rate Scheduler
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.9  # Reduce learning rate by 10% every epoch after 10 epochs

lr_scheduler = LearningRateScheduler(scheduler)

# Plot model architecture
model.summary()

# Lists of directories with studies
train_and_val_directories = [f.path for f in os.scandir(TRAIN_DATASET_PATH) if f.is_dir()]
train_and_val_directories.remove(TRAIN_DATASET_PATH + 'BraTS20_Training_355')  # Remove ill-formatted file

# Split data into training, validation, and test sets
def pathListIntoIds(dirList):
    return [dirList[i][dirList[i].rfind('/') + 1:] for i in range(len(dirList))]

train_and_test_ids = pathListIntoIds(train_and_val_directories)
train_test_ids, val_ids = train_test_split(train_and_test_ids, test_size=0.2)
train_ids, test_ids = train_test_split(train_test_ids, test_size=0.15)

# Data Generator
class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, dim=(128, 128), batch_size=1, n_channels=2, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.slices_per_step = VOLUME_SLICES // 2  # Process half slices per step

    def __len__(self):
        return int(np.floor(len(self.list_IDs) * 2 / self.batch_size))  # Adjust for half slices

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size // 2:(index + 1) * self.batch_size // 2]
        Batch_ids = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(Batch_ids)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, Batch_ids):
        """Generates data containing half slices per batch."""
        X = np.zeros((self.slices_per_step * self.batch_size, *self.dim, self.n_channels))
        Y = np.zeros((self.slices_per_step * self.batch_size, *self.dim, 4))

        for c, i in enumerate(Batch_ids):
            case_path = os.path.join(TRAIN_DATASET_PATH, i)

            # Load data
            flair = nib.load(os.path.join(case_path, f'{i}_flair.nii')).get_fdata()
            ce = nib.load(os.path.join(case_path, f'{i}_t1ce.nii')).get_fdata()
            seg = nib.load(os.path.join(case_path, f'{i}_seg.nii')).get_fdata()

            for j in range(self.slices_per_step):
                X[j + self.slices_per_step * c, :, :, 0] = cv2.resize(flair[:, :, j + VOLUME_START_AT], self.dim)
                X[j + self.slices_per_step * c, :, :, 1] = cv2.resize(ce[:, :, j + VOLUME_START_AT], self.dim)

                # Resize and process mask
                mask = cv2.resize(seg[:, :, j + VOLUME_START_AT], self.dim, interpolation=cv2.INTER_NEAREST)
                mask[mask == 4] = 3  # Convert label 4 to 3
                Y[j + self.slices_per_step * c] = tf.one_hot(mask.astype(np.int32), 4)

        return X / np.max(X), Y


# Create data generators
training_generator = DataGenerator(train_ids)
valid_generator = DataGenerator(val_ids)
test_generator = DataGenerator(test_ids)

# Show data layout
def showDataLayout():
    plt.bar(["Train", "Valid", "Test"], [len(train_ids), len(val_ids), len(test_ids)], align='center', color=['green', 'red', 'blue'])
    plt.legend()
    plt.ylabel('Number of images')
    plt.title('Data distribution')
    plt.show()

showDataLayout()

# Initialize wandb run
wandb.init(
    project="Thesis_Unet",
    entity= "xrtsoutsas-university-of-thessaly"         
)

# Callbacks
csv_logger = CSVLogger('training.log', separator=',', append=False)
callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.000001, verbose=2),
    csv_logger,
    WandbCallback(save_graph=False, save_model=False),
    lr_scheduler
]


# Post-Processing Functions
def threshold_mask(pred, threshold=0.5):
    return (pred > threshold).astype(np.uint8)

def apply_morphology(mask):
    import cv2
    kernel = np.ones((3, 3), np.uint8)
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Opening to remove small noise
    return cleaned_mask

def post_process_predictions(predictions, threshold=0.5):
    processed_preds = []
    for pred in predictions:
        thresholded = threshold_mask(pred, threshold)
        cleaned = apply_morphology(thresholded)
        processed_preds.append(cleaned)
    return np.array(processed_preds)

def plot_predictions(generator, model, num_examples=8):
    """Plot original images, predicted masks, ground truth masks, and combined view."""
    fig, axes = plt.subplots(num_examples, 4, figsize=(20, 5 * num_examples))
    
    # Randomly select examples from the generator
    for i in range(num_examples):
        index = random.randint(0, len(generator) - 1)
        X, Y_true = generator[index]  # Get data batch
        X_sample = X[0]  # First image in batch
        Y_true_sample = Y_true[0]  # First mask in batch

        # Predict mask
        Y_pred_sample = model.predict(X_sample[np.newaxis, ...])[0]
        Y_pred_sample = np.argmax(Y_pred_sample, axis=-1)
        Y_true_sample = np.argmax(Y_true_sample, axis=-1)
        
        # Combined view: overlay ground truth and predicted masks on original image
        combined = X_sample[:, :, 0].copy()  # Use the first channel (FLAIR)
        combined[Y_true_sample == 1] = 255  # Ground truth in red
        combined[Y_pred_sample == 1] = 150  # Prediction in blue
        
        # Plot original image
        axes[i, 0].imshow(X_sample[:, :, 0], cmap='gray')
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis('off')
        
        # Plot predicted mask
        axes[i, 1].imshow(Y_pred_sample, cmap='gray')
        axes[i, 1].set_title("Predicted Mask")
        axes[i, 1].axis('off')
        
        # Plot ground truth mask
        axes[i, 2].imshow(Y_true_sample, cmap='gray')
        axes[i, 2].set_title("Ground Truth Mask")
        axes[i, 2].axis('off')
        
        # Plot combined view
        axes[i, 3].imshow(combined, cmap='gray')
        axes[i, 3].set_title("Combined View")
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.show()


# Clear session and train model
K.clear_session()
# Uncomment the following lines to train the model
history = model.fit(
    training_generator, 
    epochs=35, 
    steps_per_epoch = len(training_generator),
    callbacks=callbacks, 
    validation_data=valid_generator)
model.save("model_ViT_V3.keras")

# Load trained model
#model = keras.models.load_model(r'C:\Users\Christos Tsoutsas\Desktop\University\model_per_class.h5',
 #                               custom_objects={
  #                                  'accuracy': tf.keras.metrics.MeanIoU(num_classes=4),
   #                                 "dice_coef": dice_coef,
    #                                "precision": precision,
     #                               "sensitivity": sensitivity,
      #                              "specificity": specificity,
       #                             "dice_coef_necrotic": dice_coef_necrotic,
        #                            "dice_coef_edema": dice_coef_edema,
         #                           "dice_coef_enhancing": dice_coef_enhancing,
          #                          "PatchEmbedding": PatchEmbedding,
           #                         "TransformerBlock": TransformerBlock
            #                    }, compile=False)

# Load training history
#history = pd.read_csv(r'C:\Users\Christos Tsoutsas\Desktop\University\training_per_class.log', sep=',', engine='python')

# Plot training history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
train_dice = history.history['dice_coef']
val_dice = history.history['val_dice_coef']
mean_iou = history.history['mean_io_u']
val_mean_iou = history.history['val_mean_io_u']

epoch = range(len(acc))

f, ax = plt.subplots(1, 4, figsize=(16, 8))
ax[0].plot(epoch, acc, 'b', label='Training Accuracy')
ax[0].plot(epoch, val_acc, 'r', label='Validation Accuracy')
ax[0].legend()
ax[1].plot(epoch, loss, 'b', label='Training Loss')
ax[1].plot(epoch, val_loss, 'r', label='Validation Loss')
ax[1].legend()
ax[2].plot(epoch, train_dice, 'b', label='Training dice coef')
ax[2].plot(epoch, val_dice, 'r', label='Validation dice coef')
ax[2].legend()
ax[3].plot(epoch, history['mean_io_u'], 'b', label='Training mean IOU')
ax[3].plot(epoch, history['val_mean_io_u'], 'r', label='Validation mean IOU')
ax[3].legend()
plt.show()

# Evaluate model on test data
print("Evaluate on test data")
results = model.evaluate(test_generator, batch_size=100, callbacks=callbacks)
print("test loss, test acc:", results)

# Plot predictions after training
plot_predictions(test_generator, model, num_examples=8)