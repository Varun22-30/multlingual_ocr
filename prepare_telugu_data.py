# prepare_telugu_data.py (Final Corrected Version)
# --------------------------------------------------------
# Processes a correctly structured IIIT-H dataset and handles
# tab or multi-space separated ground truth files.
# --------------------------------------------------------

import os
import shutil
import pandas as pd
from tqdm import tqdm

def prepare_dataset_final(source_base_dir, target_base_dir):
    print(f"Source Directory: {source_base_dir}")
    print(f"Target Directory: {target_base_dir}")

    all_labels_for_csv = []
    splits = ['train', 'val', 'test']

    for split in splits:
        print(f"\nProcessing split: {split}...")
        
        gt_file_path = os.path.join(source_base_dir, f'{split}_gt.txt')
        if not os.path.exists(gt_file_path):
            print(f"Warning: Ground truth file not found: {gt_file_path}. Skipping '{split}' split.")
            continue
        
        label_dict = {}
        with open(gt_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    label_dict[parts[0]] = parts[1]

        source_split_dir = os.path.join(source_base_dir, split)
        target_img_dir = os.path.join(target_base_dir, split)
        os.makedirs(target_img_dir, exist_ok=True)

        if not os.path.isdir(source_split_dir):
            print(f"Warning: Source directory for split '{split}' not found: {source_split_dir}")
            continue

        for dirpath, _, filenames in tqdm(os.walk(source_split_dir), desc=f'Processing {split} images'):
            for filename in filenames:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    relative_path = os.path.relpath(os.path.join(dirpath, filename), source_split_dir)
                    lookup_key = relative_path.replace('\\', '/')
                    label = label_dict.get(lookup_key)
                    if label is None:
                        continue

                    new_filename = lookup_key.replace('/', '_')
                    source_img_path = os.path.join(dirpath, filename)
                    target_img_path = os.path.join(target_img_dir, new_filename)
                    shutil.copyfile(source_img_path, target_img_path)
                    all_labels_for_csv.append({'filename': os.path.join(split, new_filename), 'label': label})

    if not all_labels_for_csv:
        print("\nNo data was processed. Please check your source directory and GT file paths.")
        return

    output_csv_path = os.path.join(target_base_dir, 'labels.csv')
    df = pd.DataFrame(all_labels_for_csv)
    df.to_csv(output_csv_path, index=False)
    
    print(f"\n✅ Success! Data preparation is complete.")
    print(f"Images flattened and copied to '{target_base_dir}'")
    print(f"Master labels file created at '{output_csv_path}' with {len(df)} entries.")

if __name__ == '__main__':
    source_dataset_dir = r'F:\datasets\telugu' 
    project_data_dir = './data/telugu'
    
    prepare_dataset_final(source_base_dir=source_dataset_dir, target_base_dir=project_data_dir)