# remap_labels.py

import os
import sys
import nibabel as nib
import numpy as np

def remap_labels(input_dir, output_dir):
    """
    Remap label values from 4 to 3 in all label files within the input directory.
    Save the remapped label files to the output directory, preserving the directory structure.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Traverse the input directory
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith("labels.nii.gz"):
                input_file = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                target_root = os.path.join(output_dir, relative_path)
                
                if not os.path.exists(target_root):
                    os.makedirs(target_root)
                
                output_file = os.path.join(target_root, file)
                
                print(f"Processing {input_file}...")
                
                # Load the label file
                img = nib.load(input_file)
                data = img.get_fdata().astype(np.int16)
                
                # Remap label 4 to 3
                data[data == 4] = 3
                
                # Save the remapped label file
                new_img = nib.Nifti1Image(data, img.affine, img.header)
                nib.save(new_img, output_file)
                
                print(f"Saved remapped labels to {output_file}")

    print("Label remapping completed.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python remap_labels.py /path/to/input_labels_dir /path/to/output_labels_dir")
        sys.exit(1)
    
    input_labels_dir = sys.argv[1]
    output_labels_dir = sys.argv[2]
    
    remap_labels(input_labels_dir, output_labels_dir)
