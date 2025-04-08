import os

# Define directories
TEST_DIR = "....."
OUTPUT_DIR = "....."

# Modalities
modalities = ["T1", "T1c", "T2", "FLAIR"]

# Initialize dictionaries to hold file paths
channel_files = {modality: [] for modality in modalities}

# Iterate over each subject and collect file paths
for subject in os.listdir(TEST_DIR):
    subject_dir = os.path.join(TEST_DIR, subject)
    if os.path.isdir(subject_dir):
        for modality in modalities:
            image_path = os.path.join(subject_dir, f"{modality}.nii.gz")
            if os.path.exists(image_path):
                channel_files[modality].append(image_path)
            else:
                print(f"Warning: {image_path} does not exist.")

# Write to channel list files
for modality in modalities:
    list_file_path = os.path.join(OUTPUT_DIR, f"channelsTest_{modality}.txt")
    with open(list_file_path, "w") as f:
        for path in channel_files[modality]:
            f.write(f"{path}\n")

print("Channel list files generated successfully.")
