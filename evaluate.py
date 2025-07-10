import torch
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, ToTensor
from monai.networks.nets import DenseNet121
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def evaluate_model(model_path, data_dir, batch_size=32):
    """Evaluate trained model on test data"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = DenseNet121(
        spatial_dims=2,
        in_channels=1,
        out_channels=len(os.listdir(data_dir))
    ).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Create test data loader
    classes = sorted(os.listdir(data_dir))
    file_paths = []
    labels = []
    
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            file_paths.append(os.path.join(class_dir, img_name))
            labels.append(class_idx)
    
    transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensity(),
        ToTensor()
    ])
    
    test_ds = Dataset(
        data=[{"image": img, "label": label} for img, label in zip(file_paths, labels)],
        transform=transforms
    )
    
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, num_workers=4
    )
    
    # Evaluation
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    # Generate classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=classes))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    print("Confusion matrix saved as confusion_matrix.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--data_dir', required=True, help='Path to test data')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.data_dir, args.batch_size)
