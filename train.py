import os
import torch
import numpy as np
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, 
    ScaleIntensity, RandRotate, RandFlip, RandZoom,
    ToTensor
)
from monai.networks.nets import DenseNet121
from monai.utils import set_determinism
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse

def create_data_loaders(data_dir, batch_size=32, val_split=0.2):
    """Create training and validation data loaders"""
    classes = sorted(os.listdir(data_dir))
    file_paths = []
    labels = []
    
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            file_paths.append(os.path.join(class_dir, img_name))
            labels.append(class_idx)
    
    # Split into train and validation
    train_files, val_files, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=val_split, stratify=labels, random_state=42
    )
    
    # Define transforms
    train_transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensity(),
        RandRotate(range_x=15, prob=0.5, keep_size=True),
        RandFlip(spatial_axis=0, prob=0.5),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
        ToTensor()
    ])
    
    val_transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensity(),
        ToTensor()
    ])
    
    # Create datasets
    train_ds = Dataset(
        data=[{"image": img, "label": label} for img, label in zip(train_files, train_labels)],
        transform=train_transforms
    )
    
    val_ds = Dataset(
        data=[{"image": img, "label": label} for img, label in zip(val_files, val_labels)],
        transform=val_transforms
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4
    )
    
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, num_workers=4
    )
    
    return train_loader, val_loader, classes

def train_model(train_loader, val_loader, num_classes, epochs=50, lr=1e-4):
    """Train the classification model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = DenseNet121(
        spatial_dims=2,
        in_channels=1,
        out_channels=num_classes
    ).to(device)
    
    # Loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    best_val_acc = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch["image"].to(device), batch["label"].to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = correct / total
        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved new best model")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Path to processed PNG dataset')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    
    set_determinism(seed=42)
    train_loader, val_loader, classes = create_data_loaders(args.data_dir, args.batch_size)
    model = train_model(train_loader, val_loader, len(classes), args.epochs, args.lr)
