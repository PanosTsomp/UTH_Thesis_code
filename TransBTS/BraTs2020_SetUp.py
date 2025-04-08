import os

def generate_subfolder_list(root_dir, output_txt):
    """
    Scans through the root_dir for folders matching 'BraTS20_Training_XXX'
    and writes each to output_txt.
    """
    print(f"Root directory set to: {root_dir}")
    if not os.path.isdir(root_dir):
        print("Error: The specified root directory does not exist or is not a directory.")
        return
    
    entries = os.listdir(root_dir)
    print(f"Entries in root_dir ({len(entries)} found): {entries}")

    with open(output_txt, 'w') as f:
        for entry in sorted(entries):
            print(f"Checking entry: {entry}")
            entry_path = os.path.join(root_dir, entry)

            
            if os.path.isdir(entry_path) and entry.startswith("BraTS20_Validation_"):
                print(f"  -> Matched directory: {entry}")
                f.write(entry + "\n")
            else:
                print(f"  -> Skipped: {entry}")

    print(f"Done! The list of subfolders has been written to {output_txt}")


if __name__ == "__main__":
    
    root_dir = "....."
    output_txt = "all.txt"

    generate_subfolder_list(root_dir, output_txt)
