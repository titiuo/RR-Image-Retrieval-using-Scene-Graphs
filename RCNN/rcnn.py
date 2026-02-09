import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from tqdm import tqdm
import numpy as np

# --- 1. Class Definitions ---

class RCNNDataset(Dataset):
    def __init__(self, csv_file, img_dir, obj_to_idx, attr_to_idx, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.obj_to_idx = obj_to_idx
        self.attr_to_idx = attr_to_idx
        self.transform = transform or transforms.Compose([
            transforms.Resize((227, 227)), # R-CNN requires 227x227 [cite: 2580]
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(f"{self.img_dir}/{row['image']}").convert('RGB')
        
        # Crop using [x0, y0, x1, y1] logic
        bbox = eval(row['box']) 
        cropped_img = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        
        if self.transform:
            cropped_img = self.transform(cropped_img)
            
        # Object mapping 
        obj_name = row['class']
        obj_label = self.obj_to_idx.get(obj_name, self.obj_to_idx['background'])
        
        # Attribute multi-hot mapping [cite: 2156, 2158]
        attr_vec = torch.zeros(len(self.attr_to_idx))
        current_attrs = eval(row['attrs'])
        for a in current_attrs:
            if a in self.attr_to_idx:
                attr_vec[self.attr_to_idx[a]] = 1.0
                
        return cropped_img, obj_label, attr_vec

class MultiHeadRCNN(nn.Module):
    def __init__(self, num_objects, num_attributes):
        super(MultiHeadRCNN, self).__init__()
        # Backbone is AlexNet [cite: 2580]
        self.backbone = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        feature_dim = self.backbone.classifier[6].in_features
        # Extract 4,096-D features from the fc7 layer 
        self.backbone.classifier = nn.Sequential(*list(self.backbone.classifier.children())[:-1])
        
        self.obj_head = nn.Linear(feature_dim, num_objects)
        self.attr_head = nn.Linear(feature_dim, num_attributes)

    def forward(self, x):
        features = self.backbone(x) # 4,096-dimensional features
        return self.obj_head(features), self.attr_head(features)

# --- 2. Main Execution Block ---

if __name__ == '__main__':
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_images_path = '../../sg_dataset/sg_train_images/' # Adjust as needed

    # Load mappings
    obj_df = pd.read_csv('object_classes.csv')
    attr_df = pd.read_csv('attribute_classes.csv')
    df = pd.read_csv('rcnn_training_cleaned.csv')

    obj_classes = list(obj_df['object_name'].values) + ['background'] 
    obj_to_idx = {name: i for i, name in enumerate(obj_classes)}
    attr_classes = list(attr_df['attribute_name'].values)
    attr_to_idx = {name: i for i, name in enumerate(attr_classes)}

    train_dataset = RCNNDataset('rcnn_training_cleaned.csv', train_images_path, obj_to_idx, attr_to_idx)

    # --- R-CNN Specific Sampler: 32 foreground / 96 background  ---
    # Identify indices for foreground and background
    fg_indices = df[df['class'] != 'background'].index.tolist()
    bg_indices = df[df['class'] == 'background'].index.tolist()

    # Assign weights to ensure 32:96 ratio (1:3) in each batch of 128
    weights = np.zeros(len(df))
    weights[fg_indices] = 32 / len(fg_indices)
    weights[bg_indices] = 96 / len(bg_indices)
    sampler = WeightedRandomSampler(weights, num_samples=len(df), replacement=True)

    dataloader = DataLoader(train_dataset, batch_size=128, sampler=sampler, num_workers=4)

    model = MultiHeadRCNN(num_objects=len(obj_classes), num_attributes=len(attr_classes)).to(device)
    
    # Paper uses SGD with 0.001 learning rate and 0.9 momentum [cite: 2618]
    criterion_obj = nn.CrossEntropyLoss() # Imbalance is handled by the sampler 
    criterion_attr = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(10):
        model.train()
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/10", unit="batch")
        
        for i, (images, obj_labels, attr_labels) in enumerate(pbar):
            images, obj_labels, attr_labels = images.to(device), obj_labels.to(device), attr_labels.to(device)
            
            optimizer.zero_grad()
            obj_pred, attr_pred = model(images)
            
            loss = criterion_obj(obj_pred, obj_labels) + criterion_attr(attr_pred, attr_labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 10 == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        print(f"Epoch {epoch+1} Average Loss: {running_loss / len(dataloader):.4f}")
        
    torch.save(model.state_dict(), 'rcnn_finetuned.pth')