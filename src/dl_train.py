import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import ProjectConfig
from src.data.dataset import ASVSpoofDataset
from src.models.resnet import ResNetAttention
from src.models.loss import OCSoftmax
from src.features.extract import get_transform

def train():
    cfg = ProjectConfig()
    os.makedirs(cfg.out_fold, exist_ok=True)
    
    transform = get_transform(cfg.feature_type)
    
    train_set = ASVSpoofDataset(cfg.logical_data_root, track='LA', is_train=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    
    model = ResNetAttention(enc_dim=cfg.enc_dim).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    
    criterion = nn.CrossEntropyLoss()
    loss_model = OCSoftmax(feat_dim=cfg.enc_dim, r_real=cfg.r_real, r_fake=cfg.r_fake, alpha=cfg.alpha).to(cfg.device)
    loss_optimizer = torch.optim.SGD(loss_model.parameters(), lr=cfg.learning_rate)

    print("Starting Deep Learning Training...")
    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for lfcc, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}"):
            lfcc = lfcc.unsqueeze(1).float().to(cfg.device)
            labels = labels.type(torch.LongTensor).to(cfg.device)
            
            feats, outputs = model(lfcc)
            
            oc_loss, _ = loss_model(feats, labels)
            loss = oc_loss
            
            optimizer.zero_grad()
            loss_optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(train_loader):.4f}")
        
        torch.save(model.state_dict(), os.path.join(cfg.out_fold, f'resnet_attn_{epoch}.pt'))

if __name__ == '__main__':
    train()