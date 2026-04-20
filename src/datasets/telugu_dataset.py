# src/datasets/telugu_dataset.py
# --------------------------------------------------------
# PyTorch Dataset for loading the Telugu handwriting data.
# --------------------------------------------------------

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class TeluguDataset(Dataset): #similar structure for all other datasets, just different character sets and CSV paths
    """
    Custom PyTorch Dataset for Telugu handwritten text images.
    It reads image filenames and their corresponding text labels from a CSV file.
    """
    def __init__(self, data_dir: str, csv_path: str, transform=None):
        """
        Args:
            data_dir (str): Path to the directory containing the images (e.g., 'data/telugu/train/').
            csv_path (str): Path to the CSV file with 'filename' and 'label' columns.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Load the CSV file containing image filenames and their labels
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Fetches the sample at the given index.

        Args:
            idx (int): The index of the sample.

        Returns:
            dict: A dictionary containing the image tensor and its corresponding label string.
                  {'image': image_tensor, 'label': label_string}
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image filename and label from the dataframe
        img_name = self.df.iloc[idx]['filename']
        label = self.df.iloc[idx]['label']
        
        # Construct the full image path
        img_path = os.path.join(self.data_dir, img_name)
        
        # Open the image and convert to RGB to ensure 3 channels
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: Image file not found at {img_path}. Skipping.")
            return self.__getitem__((idx + 1) % len(self))

        # Apply transformations if they exist
        if self.transform:
            image = self.transform(image)
        
        sample = {'image': image, 'label': label}
        
        return sample

# Example of how to use this dataset
if __name__ == '__main__':
    # Define the transformations needed for the ViT model
    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- Create dummy data for testing ---
    if not os.path.exists('data/telugu/train'):
        os.makedirs('data/telugu/train')
    
    # Use actual Telugu characters for the dummy labels
    dummy_labels = pd.DataFrame({
        'filename': ['sample1.png', 'sample2.png'],
        'label': ['నమస్కారం', 'తెలుగు']
    })
    dummy_labels.to_csv('data/telugu/labels.csv', index=False)
     
    Image.new('RGB', (500, 50)).save('data/telugu/train/sample1.png')
    Image.new('RGB', (400, 50)).save('data/telugu/train/sample2.png')
    # --- End of dummy data creation ---

    # Instantiate the dataset
    dataset = TeluguDataset(
        data_dir='data/telugu/train/',
        csv_path='data/telugu/labels.csv',
        transform=img_transforms
    )
    
    print(f"Dataset size: {len(dataset)}")

    # Get the first sample
    if len(dataset) > 0:
        first_sample = dataset[0]
        image_tensor = first_sample['image']
        label_text = first_sample['label']
        
        print(f"\nFirst sample's label: '{label_text}'")
        print(f"First sample's image tensor shape: {image_tensor.shape}")