import os
import pydicom
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm

def load_and_parse_metadata(csv_path):
    """Load and parse the RSNA dataset metadata"""
    df = pd.read_csv(csv_path)
    
    # Parse the ID column into components
    split_ids = df['ID'].str.split('_', expand=True)
    df['study_id'] = split_ids[1]
    df['slice_num'] = split_ids[2]
    df['hemorrhage_type'] = split_ids[3]
    df['slice_id'] = df['study_id'] + '_' + df['slice_num']
    
    # Pivot to get one row per slice with all hemorrhage types
    pivot_df = df.pivot(index='slice_id', 
                       columns='hemorrhage_type', 
                       values='Label').reset_index()
    
    return pivot_df

def process_dicom_to_png(dcm_path, output_path, size=(256, 256)):
    """Convert DICOM file to PNG with preprocessing"""
    try:
        dicom = pydicom.dcmread(dcm_path)
        img = dicom.pixel_array
        
        # Normalize and convert to uint8
        img = (img - img.min()) / (img.max() - img.min()) * 255
        img = img.astype(np.uint8)
        
        # Convert to PIL Image and resize
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize(size)
        
        # Save as PNG
        pil_img.save(output_path)
        return True
    except Exception as e:
        print(f"Error processing {dcm_path}: {str(e)}")
        return False

def create_dataset_structure(data_dir, output_dir, csv_path, max_per_class=500):
    """Create balanced dataset with specified number of samples per class"""
    classes = ['epidural', 'intraparenchymal', 'intraventricular', 
               'subarachnoid', 'subdural', 'any']
    
    # Create output directories
    for class_name in classes:
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)
    
    # Load and parse metadata
    df = load_and_parse_metadata(csv_path)
    
    # Create balanced subset
    class_counts = {c: 0 for c in classes}
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if all(count >= max_per_class for count in class_counts.values()):
            break
            
        dcm_path = os.path.join(data_dir, f"ID_{row['study_id']}_{row['slice_num']}.dcm")
        
        for class_name in classes:
            if row[class_name] == 1 and class_counts[class_name] < max_per_class:
                output_path = os.path.join(output_dir, class_name, f"{row['slice_id']}.png")
                if process_dicom_to_png(dcm_path, output_path):
                    class_counts[class_name] += 1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dicom_dir', required=True, help='Path to DICOM files')
    parser.add_argument('--csv_path', required=True, help='Path to metadata CSV')
    parser.add_argument('--output_dir', required=True, help='Output directory for PNGs')
    parser.add_argument('--max_per_class', type=int, default=500, help='Max images per class')
    args = parser.parse_args()
    
    create_dataset_structure(args.dicom_dir, args.output_dir, args.csv_path, args.max_per_class)
