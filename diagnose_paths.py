# diagnose_paths.py
# --------------------------------------------------------
# A dedicated script to diagnose path mismatch issues.
# --------------------------------------------------------
import os

def run_diagnostics(source_base_dir):
    """
    Finds the first image and compares its generated path with the GT file paths.
    """
    print(f"--- Running Path Diagnostics on: '{source_base_dir}' ---")

    split = 'train'
    gt_file_path = os.path.join(source_base_dir, f'{split}_gt.txt')
    source_split_dir = os.path.join(source_base_dir, split)

    # --- 1. Basic Existence Checks ---
    if not os.path.exists(gt_file_path):
        print(f"❌ ERROR: Ground truth file NOT FOUND at: '{gt_file_path}'")
        return
    if not os.path.isdir(source_split_dir):
        print(f"❌ ERROR: Image folder NOT FOUND at: '{source_split_dir}'")
        return
    
    print("✅ GT file and Image folder found.")

    # --- 2. Load all keys from the GT file ---
    label_dict = {}
    with open(gt_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                label_dict[parts[0]] = parts[1]
    
    if not label_dict:
        print("❌ ERROR: Failed to read any valid data from the GT file. It might be empty or in an unexpected format.")
        return
    
    print(f"✅ Successfully loaded {len(label_dict)} labels from the GT file.")

    # --- 3. Find the first image and generate its key ---
    for dirpath, _, filenames in os.walk(source_split_dir):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # This is the first image file we found.
                full_path = os.path.join(dirpath, filename)
                generated_key = os.path.relpath(full_path, source_split_dir).replace('\\', '/')
                
                print("\n--- Match Analysis ---")
                print(f"1. Path generated from your folder structure:\n   '{generated_key}'")
                
                # Check if this key exists in our dictionary of labels
                match_found = generated_key in label_dict
                print(f"\n2. Does this key exist in the GT file? -> {match_found}")

                if not match_found:
                    print("\n3. Let's compare it to some actual keys from your GT file:")
                    # Print the first 5 keys from the GT file for comparison
                    for i, key in enumerate(label_dict.keys()):
                        if i >= 5:
                            break
                        print(f"   Sample Key from GT file: '{key}'")
                
                print("\n--- Diagnostics Complete ---")
                return # Stop after analyzing the first image

    print("❌ ERROR: Could not find a single image file to analyze in the specified directory.")

if __name__ == '__main__':
    # Make sure this path is correct
    source_dataset_dir = r'F:\datasets\telugu'
    run_diagnostics(source_base_dir=source_dataset_dir)