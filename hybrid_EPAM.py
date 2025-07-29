import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from utils_HelperFunction import extract_eigencam_maps, save_attention_visualizations

# EigenCAM Attention Module
class EigenCAMAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.in_channels = in_channels
        self.reduction_channels = max(in_channels // reduction_ratio, 8)
        
        # Spatial attention branch
        self.spatial_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        
        # Channel attention branch (similar to SE but using EigenDecomposition logic)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, self.reduction_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(self.reduction_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
        # Projection for eigenvalue computation
        self.eigen_proj = nn.Conv2d(in_channels, self.reduction_channels, kernel_size=1)
        
    def _compute_eigen_attention(self, x):
        batch_size, channels, height, width = x.size()
        
        # Project features to lower dimension for efficiency
        feat = self.eigen_proj(x)
        feat = feat.view(batch_size, self.reduction_channels, -1)  # B x C' x (HW)
        
        # Compute covariance matrix
        feat_mean = torch.mean(feat, dim=2, keepdim=True)
        feat_centered = feat - feat_mean
        cov = torch.bmm(feat_centered, feat_centered.transpose(1, 2))
        cov = cov / (height * width - 1) + 1e-5 * torch.eye(self.reduction_channels, device=x.device)
        
        # Approximation of eigenvalue decomposition using power iteration method
        # (for efficiency compared to full eigendecomposition)
        v = torch.randn(batch_size, self.reduction_channels, 1, device=x.device)
        v = v / torch.norm(v, dim=1, keepdim=True)
        
        # Power iteration (5 iterations is typically sufficient)
        for _ in range(5):
            v = torch.bmm(cov, v)
            v = v / (torch.norm(v, dim=1, keepdim=True) + 1e-10)
        
        # v now approximates the principal eigenvector
        # Project features onto this eigenvector to get attention
        attention = torch.bmm(v.transpose(1, 2), feat_centered)
        attention = attention.view(batch_size, 1, height, width)
        
        # Normalize to [0, 1]
        attention = attention - attention.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        attention = attention / (attention.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0] + 1e-10)
        
        return attention
    
    def forward(self, x):
        # Channel attention (SE-like)
        channel_att = self.avg_pool(x)
        channel_att = self.fc1(channel_att)
        channel_att = self.relu(channel_att)
        channel_att = self.fc2(channel_att)
        channel_att = self.sigmoid(channel_att)
        
        # EigenCAM attention
        eigen_att = self._compute_eigen_attention(x)
        eigen_att = self.sigmoid(eigen_att)
        
        # Spatial attention
        spatial_att = self.spatial_conv(x)
        spatial_att = self.sigmoid(spatial_att)
        
        # Combine attentions
        att = eigen_att * spatial_att
        
        # Apply attention
        x_att = x * channel_att * att
        
        return x_att + x  # Residual connection

# Global Response Normalization
class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=-1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

# DropPath from timm
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

# Modified ConvNeXtV2 Block with EigenCAM Attention
class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim, drop_path=0., use_eigen_attention=False):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # Add EigenCAM attention
        self.use_eigen_attention = use_eigen_attention
        if use_eigen_attention:
            self.eigen_attention = EigenCAMAttention(dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        
        # Apply drop path
        x = input + self.drop_path(x)
        
        # Apply EigenCAM attention if enabled
        if self.use_eigen_attention:
            x = self.eigen_attention(x)
            
        return x

# MobileViTv2 Transformer Block
class MobileViTv2SeparableAttention(nn.Module):
    def __init__(self, in_channels, out_channels, attn_unit_dim=8, ffn_multiplier=2.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        ffn_dims = int(in_channels * ffn_multiplier)
        
        # Local representation
        self.local_rep = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )
        
        # Global representation
        self.global_rep = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        
        # Separable self-attention
        attn_dim = max(attn_unit_dim, int(in_channels // 16))
        attn_dim = attn_dim if attn_dim % attn_unit_dim == 0 else (attn_dim // attn_unit_dim) * attn_unit_dim
        
        self.attn_layer = nn.Sequential(
            nn.Conv2d(in_channels, attn_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(attn_dim),
            nn.SiLU(),
            nn.Conv2d(attn_dim, attn_dim, kernel_size=3, padding=1, groups=attn_dim, bias=False),
            nn.BatchNorm2d(attn_dim),
            nn.Conv2d(attn_dim, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Conv2d(in_channels, ffn_dims, kernel_size=1, bias=False),
            nn.BatchNorm2d(ffn_dims),
            nn.SiLU(),
            nn.Conv2d(ffn_dims, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.out_proj = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
    
    def forward(self, x):
        # Local representation
        local_feat = self.local_rep(x)
        
        # Global representation
        global_feat = self.global_rep(local_feat)
        
        # Self-attention
        attn_feat = self.attn_layer(global_feat)
        attn_feat = attn_feat + global_feat
        
        # FFN
        ffn_feat = self.ffn(attn_feat)
        
        # Combine features
        output = self.out_proj(torch.cat([attn_feat, ffn_feat], dim=1))
        return output

# Downsampling layer
class DownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=stride, stride=stride)
        self.norm = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        return self.norm(self.conv(x))

# Improved Hybrid model with EigenCAM applied to ConvNeXtV2 stages
class ImprovedHybridModel(nn.Module):
    def __init__(self, num_classes=3, input_channels=3, use_eigen_attention=True):
        super().__init__()
        self.use_eigen_attention = use_eigen_attention
        
        # Initial downsampling
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=4),
            nn.BatchNorm2d(64)
        )
        
        # Stage 1 (ConvNeXtV2 with EigenCAM) - C1 channels after downsampling by 4
        C1 = 96
        self.stage1 = nn.ModuleList([
            ConvNeXtV2Block(dim=C1, drop_path=0.1, use_eigen_attention=use_eigen_attention) 
            for _ in range(3)
        ])
        self.stage1_conv = nn.Conv2d(64, C1, kernel_size=1)
        self.downsample1 = DownsampleLayer(C1, C1*2)
        
        # Stage 2 (ConvNeXtV2 with EigenCAM) - C2 channels after downsampling by 8
        C2 = 192
        self.stage2 = nn.ModuleList([
            ConvNeXtV2Block(dim=C2, drop_path=0.1, use_eigen_attention=use_eigen_attention) 
            for _ in range(3)
        ])
        self.downsample2 = DownsampleLayer(C2, C2*2)
        
        # Stage 3 (MobileViTv2) - C3 channels after downsampling by 16
        C3 = 384
        self.stage3 = nn.ModuleList([
            MobileViTv2SeparableAttention(in_channels=C3, out_channels=C3) 
            for _ in range(3)
        ])
        self.downsample3 = DownsampleLayer(C3, C3*2)
        
        # Stage 4 (MobileViTv2) - C4 channels after downsampling by 32
        C4 = 768
        self.stage4 = nn.ModuleList([
            MobileViTv2SeparableAttention(in_channels=C4, out_channels=C4) 
            for _ in range(3)
        ])
        
        # Add final EigenCAM attention
        if use_eigen_attention:
            self.final_attention = EigenCAMAttention(C4)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(C4, 1280),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )
        
        # Store feature maps and attention maps for visualization
        self.feature_maps = {}
        self.attention_maps = {}
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, return_attention=False):
        # Initial shape: (B, 3, H, W)
        self.feature_maps.clear()
        self.attention_maps.clear()
        
        # Downsample to (B, 64, H/4, W/4)
        x = self.stem(x)
        
        # Stage 1: (B, C1, H/4, W/4) with EigenCAM attention
        x = self.stage1_conv(x)
        for i, block in enumerate(self.stage1):
            x = block(x)
            # Save attention maps from the last block of each stage
            if i == len(self.stage1) - 1 and self.use_eigen_attention:
                if hasattr(block, 'eigen_attention'):
                    attention_diff = x - block.eigen_attention(x)
                    attention_map = torch.sum(torch.abs(attention_diff), dim=1, keepdim=True)
                    self.attention_maps['stage1'] = attention_map.detach()
        self.feature_maps['stage1'] = x.detach()
        
        # Downsample to (B, C2, H/8, W/8)
        x = self.downsample1(x)
        
        # Stage 2: (B, C2, H/8, W/8) with EigenCAM attention
        for i, block in enumerate(self.stage2):
            x = block(x)
            # Save attention maps from the last block of each stage
            if i == len(self.stage2) - 1 and self.use_eigen_attention:
                if hasattr(block, 'eigen_attention'):
                    attention_diff = x - block.eigen_attention(x)
                    attention_map = torch.sum(torch.abs(attention_diff), dim=1, keepdim=True)
                    self.attention_maps['stage2'] = attention_map.detach()
        self.feature_maps['stage2'] = x.detach()
        
        # Downsample to (B, C3, H/16, W/16)
        x = self.downsample2(x)
        
        # Stage 3: (B, C3, H/16, W/16) MobileViT blocks
        for i, block in enumerate(self.stage3):
            x = block(x)
            if i == len(self.stage3) - 1:
                self.feature_maps['stage3'] = x.detach()
        
        # Downsample to (B, C4, H/32, W/32)
        x = self.downsample3(x)
        
        # Stage 4: (B, C4, H/32, W/32) MobileViT blocks
        for i, block in enumerate(self.stage4):
            x = block(x)
            if i == len(self.stage4) - 1:
                self.feature_maps['stage4'] = x.detach()
        
        # Apply final EigenCAM attention
        if self.use_eigen_attention:
            x_with_attention = self.final_attention(x)
            # Store the attention map for visualization
            attention_diff = x_with_attention - x
            attention_map = torch.sum(torch.abs(attention_diff), dim=1, keepdim=True)
            self.attention_maps['final'] = attention_map.detach()
            x = x_with_attention
        
        # Classification head
        x_pooled = self.avgpool(x)
        x_flat = torch.flatten(x_pooled, 1)
        output = self.classifier(x_flat)
        
        if return_attention:
            return output, self.feature_maps, self.attention_maps
        return output

# Training code with EigenCAM visualization
def train_model(model, train_loader, val_loader, num_epochs=50, save_attention_maps=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Track the best model
    best_val_acc = 0.0
    best_model_wts = None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            if save_attention_maps and epoch % 5 == 0:  # Save attention maps every 5 epochs
                outputs, _, _ = model(inputs, return_attention=True)
            else:
                outputs = model(inputs)
                
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                if save_attention_maps and epoch % 5 == 0:  # Save attention maps every 5 epochs
                    outputs, feature_maps, attention_maps = model(inputs, return_attention=True)
                    
                    # Save the first batch for visualization
                    if total == 0 and save_attention_maps:
                        save_attention_visualizations(inputs, feature_maps, attention_maps, epoch)
                else:
                    outputs = model(inputs)
                
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_acc = 100. * correct / total
        scheduler.step()
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = model.state_dict().copy()
            torch.save(best_model_wts, 'best_improved_hybrid_eigen_model.pth')
        
        print(f'Epoch: {epoch+1}/{num_epochs} | Train Loss: {train_loss/len(train_loader):.4f} | '
              f'Train Acc: {train_acc:.2f}% | Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.2f}%')
    
    # Load the best model weights
    model.load_state_dict(best_model_wts)
    return model



# Example usage
def main():
    # Create the model
    model = ImprovedHybridModel(num_classes=3, use_eigen_attention=True)
    
    # Print model summary
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Example input shape: 224x224x3 image
    example_input = torch.randn(1, 3, 224, 224)
    output = model(example_input)
    print(f"Output shape: {output.shape}")
    
    # Extract and visualize EigenCAM maps (for demonstration)
    attention_maps, feature_maps, predicted_class = extract_eigencam_maps(model, example_input)
    print(f"Predicted class: {predicted_class.item()}")
    print(f"Generated attention maps for layers: {list(attention_maps.keys())}")
    print(f"Generated feature maps for layers: {list(feature_maps.keys())}")

if __name__ == "__main__":
    main()