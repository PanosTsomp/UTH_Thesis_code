#!/bin/bash

# =============================
# DeepMedic Data Preparation Script
# =============================

# Exit immediately if a command exits with a non-zero status
set -e

# -----------------------------
# 1. Define Directory Paths
# -----------------------------

# Source directory containing the original BraTS training data
SOURCE_DIR="....."

# Target base directory for the new data splits
TARGET_BASE_DIR="...."

# Directory to store channel list files
CHANNELS_DIR="$TARGET_BASE_DIR/channels"

# -----------------------------
# 2. Create Necessary Directories
# -----------------------------

echo "Creating target directories..." >&2

mkdir -p "$TARGET_BASE_DIR/train"
mkdir -p "$TARGET_BASE_DIR/validation"
mkdir -p "$TARGET_BASE_DIR/test"
mkdir -p "$CHANNELS_DIR"

echo "Target directories created." >&2

# -----------------------------
# 3. Gather and Shuffle Subject Directories
# -----------------------------

echo "Gathering subject directories..." >&2

# Array of all subject directories
subjects=("$SOURCE_DIR"/BraTS20_Training_*)

# Total number of subjects
total_subjects=${#subjects[@]}

echo "Total subjects found: $total_subjects" >&2

# Shuffle the subjects array
shuffled_subjects=($(printf "%s\n" "${subjects[@]}" | shuf))

# -----------------------------
# 4. Calculate Split Sizes
# -----------------------------

# Desired split percentages
train_percent=70
validation_percent=10
test_percent=20

# Calculate number of subjects for each split
num_train=$((total_subjects * train_percent / 100))
num_validation=$((total_subjects * validation_percent / 100))
num_test=$((total_subjects - num_train - num_validation))

echo "Data split:" >&2
echo "Training: $num_train subjects ($train_percent%)" >&2
echo "Validation: $num_validation subjects ($validation_percent%)" >&2
echo "Test: $num_test subjects ($test_percent%)" >&2

# -----------------------------
# 5. Split the Subjects
# -----------------------------

# Extract subsets
train_subjects=("${shuffled_subjects[@]:0:num_train}")
validation_subjects=("${shuffled_subjects[@]:num_train:num_validation}")
test_subjects=("${shuffled_subjects[@]:num_train+num_validation}")

# -----------------------------
# 6. Function to Copy and Rename Files
# -----------------------------

copy_and_rename() {
    local set_name=$1
    shift
    local subjects=("$@")
    local target_dir="$TARGET_BASE_DIR/$set_name"

    for subject_path in "${subjects[@]}"; do
        subject_id=$(basename "$subject_path")  # e.g., BraTS20_Training_001
        echo "Processing $subject_id for $set_name set..." >&2

        # Create subject directory in the target set
        mkdir -p "$target_dir/$subject_id"

        # Define source files
        t1_file=$(find "$subject_path" -maxdepth 1 -iname "*_t1.nii*" | head -n 1)
        t1c_file=$(find "$subject_path" -maxdepth 1 -iname "*_t1ce.nii*" | head -n 1)
        t2_file=$(find "$subject_path" -maxdepth 1 -iname "*_t2.nii*" | head -n 1)
        flair_file=$(find "$subject_path" -maxdepth 1 -iname "*_flair.nii*" | head -n 1)
        seg_file=$(find "$subject_path" -maxdepth 1 -iname "*_seg.nii*" | head -n 1)

        # Check if all modality files exist
        if [ ! -f "$t1_file" ] || [ ! -f "$t1c_file" ] || [ ! -f "$t2_file" ] || [ ! -f "$flair_file" ]; then
            echo "Error: Missing modality files for $subject_id. Skipping this subject." >&2
            continue
        fi

        # Function to copy and compress if needed
        copy_and_compress() {
            local src_file=$1
            local dest_file=$2

            if [[ "$src_file" == *.nii.gz ]]; then
                # Source is already compressed; copy directly
                cp "$src_file" "$dest_file" || { echo "Error copying $src_file to $dest_file" >&2; exit 1; }
            elif [[ "$src_file" == *.nii ]]; then
                # Source is uncompressed; copy and compress
                cp "$src_file" "$dest_file.temp" || { echo "Error copying $src_file to $dest_file.temp" >&2; exit 1; }
                gzip "$dest_file.temp" || { echo "Error compressing $dest_file.temp" >&2; exit 1; }
                mv "$dest_file.temp.gz" "$dest_file" || { echo "Error renaming $dest_file.temp.gz to $dest_file" >&2; exit 1; }
            else
                echo "Warning: Unrecognized file extension for $src_file. Skipping." >&2
            fi
        }

        # Copy and compress modality files
        copy_and_compress "$t1_file" "$target_dir/$subject_id/T1.nii.gz"
        copy_and_compress "$t1c_file" "$target_dir/$subject_id/T1c.nii.gz"
        copy_and_compress "$t2_file" "$target_dir/$subject_id/T2.nii.gz"
        copy_and_compress "$flair_file" "$target_dir/$subject_id/FLAIR.nii.gz"

        # Copy and rename the label file (now including test set)
        if [ -f "$seg_file" ]; then
            if [[ "$seg_file" == *.nii.gz ]]; then
                cp "$seg_file" "$target_dir/$subject_id/labels.nii.gz" || { echo "Error copying $seg_file to $target_dir/$subject_id/labels.nii.gz" >&2; exit 1; }
            elif [[ "$seg_file" == *.nii ]]; then
                cp "$seg_file" "$target_dir/$subject_id/labels.nii.gz.temp" || { echo "Error copying $seg_file to $target_dir/$subject_id/labels.nii.gz.temp" >&2; exit 1; }
                gzip "$target_dir/$subject_id/labels.nii.gz.temp" || { echo "Error compressing $target_dir/$subject_id/labels.nii.gz.temp" >&2; exit 1; }
                mv "$target_dir/$subject_id/labels.nii.gz.temp.gz" "$target_dir/$subject_id/labels.nii.gz" || { echo "Error renaming $target_dir/$subject_id/labels.nii.gz.temp.gz to $target_dir/$subject_id/labels.nii.gz" >&2; exit 1; }
            else
                echo "Warning: Unrecognized file extension for $seg_file. Skipping." >&2
            fi
        else
            echo "Warning: Segmentation file not found for $subject_id in $set_name set." >&2
        fi
    done
}

# -----------------------------
# 7. Copy and Rename Files for Each Set
# -----------------------------

echo "Copying and renaming files for training set..." >&2
copy_and_rename "train" "${train_subjects[@]}"

echo "Copying and renaming files for validation set..." >&2
copy_and_rename "validation" "${validation_subjects[@]}"

echo "Copying and renaming files for test set..." >&2
copy_and_rename "test" "${test_subjects[@]}"

echo "Data splitting and organization complete." >&2

# -----------------------------
# 8. Function to Create Ground Truth Label List Files
# -----------------------------

create_gt_labels_list() {
    local set_name=$1
    local target_dir="$TARGET_BASE_DIR/$set_name"
    local list_dir="$CHANNELS_DIR"

    # Create labels list for all sets (including test)
    local list_file="$list_dir/gtLabels${set_name^}.txt"

    echo "Creating ground truth labels list file for $set_name set: $list_file" >&2

    # Find all labels.nii.gz files in the set and write their absolute paths to the list file
    find "$target_dir" -type f -iname "labels.nii.gz" | sort > "$list_file"

    echo "Ground truth labels list file created with $(wc -l < "$list_file") entries." >&2
}

# -----------------------------
# 9. Create Ground Truth Label List Files for Each Set
# -----------------------------

echo "Creating ground truth labels list files..." >&2

create_gt_labels_list "train"
create_gt_labels_list "validation"
create_gt_labels_list "test"  # Added to include test set

echo "Ground truth labels list files created successfully." >&2

# -----------------------------
# 10. Function to Create Channel List Files
# -----------------------------

create_channel_lists() {
    local set_name=$1
    local target_dir="$TARGET_BASE_DIR/$set_name"
    local list_dir="$CHANNELS_DIR"

    # Define modalities
    local modalities=("T1" "T1c" "T2" "FLAIR")

    for mod in "${modalities[@]}"; do
        # Define the list file name (e.g., channelsTrain_T1.txt)
        local list_file="$list_dir/channels${set_name^}_${mod}.txt"

        echo "Creating list file for $set_name set, modality $mod: $list_file" >&2

        # Find all modality files in the set and write their absolute paths to the list file
        find "$target_dir" -type f -iname "${mod}.nii.gz" | sort > "$list_file"

        echo "List file created with $(wc -l < "$list_file") entries." >&2
    done
}

# -----------------------------
# 11. Create Channel List Files for Each Set
# -----------------------------

echo "Creating channel list files..." >&2

create_channel_lists "train"
create_channel_lists "validation"
create_channel_lists "test"

echo "Channel list files created successfully." >&2

# -----------------------------
# 12. Function to Create Names for Predictions Validation List File
# -----------------------------

create_names_for_predictions_val_list() {
    local set_name="validation"
    local target_dir="$TARGET_BASE_DIR/$set_name"
    local list_dir="$CHANNELS_DIR"
    local list_file="$list_dir/namesForPredictionsPerCaseVal.txt"

    echo "Creating namesForPredictionsPerCaseVal.txt for $set_name set: $list_file" >&2

    # Extract the base names of each subject directory in the validation set
    find "$target_dir" -type d -mindepth 1 -maxdepth 1 | sort | awk -F'/' '{print $NF}' > "$list_file"

    echo "namesForPredictionsPerCaseVal.txt created with $(wc -l < "$list_file") entries." >&2
    cat "$list_file"
}

# -----------------------------
# 13. Create Names for Predictions Validation List File
# -----------------------------

echo "Creating namesForPredictionsPerCaseVal.txt..." >&2

create_names_for_predictions_val_list

echo "namesForPredictionsPerCaseVal.txt created successfully." >&2

# -----------------------------
# 14. Summary
# -----------------------------

echo "Data preparation completed successfully." >&2
echo "Train, validation, and test sets have been created with corresponding channel list files and ground truth labels." >&2
echo "You can find the channel list files in: $CHANNELS_DIR" >&2
