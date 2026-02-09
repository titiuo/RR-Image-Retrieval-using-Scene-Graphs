import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from tqdm import tqdm
import numpy as np
import ast # Safer and faster than eval()

class RCNNDataset(Dataset):
    def __init__(self, csv_file, img_dir, obj_to_idx, attr_to_idx, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.obj_to_idx = obj_to_idx
        self.attr_to_idx = attr_to_idx
        
        # PRE-PARSE: Faster than calling eval() in __getitem__
        self.df['box'] = self.df['box'].apply(ast.literal_eval)
        self.df['attrs'] = self.df['attrs'].apply(ast.literal_eval)
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((227, 227)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(f"{self.img_dir}/{row['image']}").convert('RGB')
        
        # Crop using pre-parsed list
        bbox = row['box'] 
        cropped_img = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        
        if self.transform:
            cropped_img = self.transform(cropped_img)
            
        obj_name = row['class']
        obj_label = self.obj_to_idx.get(obj_name, self.obj_to_idx['background'])
        
        # Multi-hot attributes [cite: 285]
        attr_vec = torch.zeros(len(self.attr_to_idx))
        for a in row['attrs']:
            if a in self.attr_to_idx:
                attr_vec[self.attr_to_idx[a]] = 1.0
                
        return cropped_img, obj_label, attr_vec

class MultiHeadRCNN(nn.Module):
    def __init__(self, num_objects, num_attributes):
        super(MultiHeadRCNN, self).__init__()
        # Backbone is AlexNet [cite: 648, 708]
        self.backbone = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        
        # OPTIONAL: Freeze conv layers if you only want to train the heads
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        
        feature_dim = self.backbone.classifier[6].in_features
        # Extract features from fc7 layer [cite: 708, 709]
        self.backbone.classifier = nn.Sequential(*list(self.backbone.classifier.children())[:-1])
        
        self.obj_head = nn.Linear(feature_dim, num_objects)
        self.attr_head = nn.Linear(feature_dim, num_attributes)

    def forward(self, x):
        features = self.backbone(x) # 4,096-D feature vector [cite: 708]
        return self.obj_head(features), self.attr_head(features)

if __name__ == '__main__':
    # Use MPS for Mac
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_images_path = '../../sg_dataset/sg_train_images/' 

    # Load mappings
    obj_df = pd.read_csv('object_classes.csv')
    attr_df = pd.read_csv('attribute_classes.csv')
    df_labels = pd.read_csv('rcnn_training_cleaned.csv')

    obj_classes = list(obj_df['object_name'].values) + ['background'] 
    obj_to_idx = {name: i for i, name in enumerate(obj_classes)}
    attr_to_idx = {name: i for i, name in enumerate(attr_df['attribute_name'].values)}

    train_dataset = RCNNDataset('rcnn_training_cleaned.csv', train_images_path, obj_to_idx, attr_to_idx)

    # R-CNN Batch Sampler: 32 pos / 96 neg [cite: 748]
    fg_indices = df_labels[df_labels['class'] != 'background'].index.tolist()
    bg_indices = df_labels[df_labels['class'] == 'background'].index.tolist()

    weights = np.zeros(len(df_labels))
    weights[fg_indices] = 32 / len(fg_indices)
    weights[bg_indices] = 96 / len(bg_indices)
    sampler = WeightedRandomSampler(weights, num_samples=len(df_labels), replacement=True)

    dataloader = DataLoader(train_dataset, batch_size=128, sampler=sampler, num_workers=4, pin_memory=True)

    model = MultiHeadRCNN(num_objects=len(obj_classes), num_attributes=len(attr_to_idx)).to(device)
    
    # Paper uses SGD [cite: 742, 747]
    criterion_obj = nn.CrossEntropyLoss() 
    criterion_attr = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(10):
        model.train()
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/10")
        
        for images, obj_labels, attr_labels in pbar:
            images, obj_labels, attr_labels = images.to(device), obj_labels.to(device), attr_labels.to(device)
            
            optimizer.zero_grad()
            obj_pred, attr_pred = model(images)
            
            # Multi-task loss for objects and attributes [cite: 285, 287]
            loss = criterion_obj(obj_pred, obj_labels) + criterion_attr(attr_pred, attr_labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        print(f"Epoch {epoch+1} Avg Loss: {running_loss / len(dataloader):.4f}")
        
    torch.save(model.state_dict(), 'rcnn_finetuned.pth')