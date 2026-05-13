import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from mamba_ssm import Mamba

# ==========================================
# 1. ARCHITECTURE DEFINITIONS
# ==========================================

class PureMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        return x + residual

class MultiScaleMambaEncoder(nn.Module):
    def __init__(self, input_dim=1024, embed_dim=256, num_levels=6, mamba_layers_per_level=2):
        super().__init__()
        self.feature_embedding = nn.Sequential(
            nn.Conv1d(input_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.levels = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        for i in range(num_levels):
            level_encoder = nn.Sequential(*[
                PureMambaBlock(d_model=embed_dim) for _ in range(mamba_layers_per_level)
            ])
            self.levels.append(level_encoder)
            if i < num_levels - 1:
                self.downsamples.append(
                    nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
                )

    def forward(self, x):
        x = self.feature_embedding(x)
        x = x.permute(0, 2, 1) # Mamba needs (Batch, Seq_Len, Dim)
        
        multi_scale_features = []
        for i in range(len(self.levels)):
            x = self.levels[i](x)
            multi_scale_features.append(x)
            if i < len(self.downsamples):
                x = x.permute(0, 2, 1)
                x = self.downsamples[i](x)
                x = x.permute(0, 2, 1)
        return multi_scale_features

class ActionLocalizationHead(nn.Module):
    def __init__(self, embed_dim=256, num_classes=20):
        """
        Shared head across all multi-scale levels (as per paper).
        Predicts classification scores and boundary distances.
        """
        super().__init__()
        self.num_classes = num_classes
        
        # Classification Branch
        self.cls_tower = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1), nn.ReLU()
        )
        self.cls_logits = nn.Conv1d(embed_dim, num_classes, kernel_size=3, padding=1)
        
        # Regression Branch (predicts distance to start and end boundaries)
        self.reg_tower = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1), nn.ReLU()
        )
        self.reg_pred = nn.Conv1d(embed_dim, 2, kernel_size=3, padding=1)

    def forward(self, x):
        # x is (Batch, Seq_Len, Dim), convert back to Conv1D format
        x = x.permute(0, 2, 1) 
        
        cls_feat = self.cls_tower(x)
        reg_feat = self.reg_tower(x)
        
        # Output shapes: (Batch, Num_Classes, Seq_Len), (Batch, 2, Seq_Len)
        return self.cls_logits(cls_feat), F.relu(self.reg_pred(reg_feat))

class MambaTAL(nn.Module):
    def __init__(self, input_dim=1024, embed_dim=256, num_classes=20, num_levels=6):
        super().__init__()
        self.encoder = MultiScaleMambaEncoder(input_dim, embed_dim, num_levels)
        self.head = ActionLocalizationHead(embed_dim, num_classes)

    def forward(self, x):
        multi_scale_feats = self.encoder(x)
        outputs = []
        for feat in multi_scale_feats:
            cls_out, reg_out = self.head(feat)
            outputs.append({"cls": cls_out, "reg": reg_out})
        return outputs

# ==========================================
# 2. LOSS FUNCTIONS (As specified in paper)
# ==========================================

def focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    """Standard Focal Loss for classification."""
    bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1-pt)**gamma * bce_loss
    return focal_loss.mean()

def distance_iou_loss_1d(pred_distances, target_distances):
    """1D DIoU Loss for boundary distance regression."""
    pred_left, pred_right = pred_distances[:, 0, :], pred_distances[:, 1, :]
    tgt_left, tgt_right = target_distances[:, 0, :], target_distances[:, 1, :]
    
    intersect_left = torch.min(pred_left, tgt_left)
    intersect_right = torch.min(pred_right, tgt_right)
    intersection = F.relu(intersect_left + intersect_right)
    
    union = (pred_left + pred_right) + (tgt_left + tgt_right) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    # Distance between centers
    center_pred = (pred_right - pred_left) / 2
    center_tgt = (tgt_right - tgt_left) / 2
    center_distance = (center_pred - center_tgt) ** 2
    
    # Diagonal length of the smallest enclosing box
    enclose_left = torch.max(pred_left, tgt_left)
    enclose_right = torch.max(pred_right, tgt_right)
    enclose_len = F.relu(enclose_left + enclose_right)
    
    diou = iou - (center_distance / (enclose_len**2 + 1e-6))
    return (1 - diou).mean()

# ==========================================
# 3. DUMMY DATASET & TRAINING LOOP
# ==========================================

class THUMOS14Dummy(Dataset):
    def __init__(self, num_samples=100, seq_len=2304, input_dim=1024, num_classes=20):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # I3D Features
        features = torch.randn(self.input_dim, self.seq_len)
        # Dummy Targets for the top-level resolution
        target_cls = torch.zeros(self.num_classes, self.seq_len)
        target_reg = torch.abs(torch.randn(2, self.seq_len)) 
        return features, target_cls, target_reg

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Initialize Model, Data, and Optimizer
    model = MambaTAL(input_dim=1024, embed_dim=256, num_classes=20).to(device)
    dataset = THUMOS14Dummy(num_samples=16) # Small dataset for testing
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Paper mentions Adam optimizer with a warmup strategy
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (features, tgt_cls, tgt_reg) in enumerate(dataloader):
            features, tgt_cls, tgt_reg = features.to(device), tgt_cls.to(device), tgt_reg.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            
            # For simplicity in this dummy script, we only calculate loss on the first scale level (highest resolution).
            # In full TAL training, you map ground truth to all pyramid levels.
            pred_cls = outputs[0]['cls']
            pred_reg = outputs[0]['reg']
            
            # Balance coefficient as mentioned in the paper
            loss_cls = focal_loss(pred_cls, tgt_cls)
            loss_reg = distance_iou_loss_1d(pred_reg, tgt_reg)
            loss = loss_cls + (2.0 * loss_reg) 
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(dataloader):.4f}")
    
    print("Dummy training complete! Architecture is ready for real data.")

if __name__ == "__main__":
    train()