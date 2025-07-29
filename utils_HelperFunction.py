import torch


# Function to save attention map visualizations
def save_attention_visualizations(inputs, feature_maps, attention_maps, epoch):
    import matplotlib.pyplot as plt
    
    # Create a directory to save visualizations if it doesn't exist
    import os
    os.makedirs('visualizations', exist_ok=True)
    
    # Take the first image from the batch
    img = inputs[0].cpu().permute(1, 2, 0).numpy()
    # Normalize to [0, 1] for visualization
    img = (img - img.min()) / (img.max() - img.min())
    
    plt.figure(figsize=(15, 10))
    
    # Plot the original image
    plt.subplot(2, 3, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot attention maps from ConvNeXt stages
    if 'stage1' in attention_maps:
        att_map = attention_maps['stage1'][0, 0].cpu().numpy()
        att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)
        
        plt.subplot(2, 3, 2)
        plt.imshow(att_map, cmap='jet')
        plt.title('Stage 1 EigenCAM Attention')
        plt.axis('off')
    
    if 'stage2' in attention_maps:
        att_map = attention_maps['stage2'][0, 0].cpu().numpy()
        att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)
        
        plt.subplot(2, 3, 3)
        plt.imshow(att_map, cmap='jet')
        plt.title('Stage 2 EigenCAM Attention')
        plt.axis('off')
    
    # Plot feature maps from later stages
    if 'stage3' in feature_maps:
        feat_map = feature_maps['stage3'][0].mean(dim=0).cpu().numpy()
        feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min() + 1e-8)
        
        plt.subplot(2, 3, 4)
        plt.imshow(feat_map, cmap='viridis')
        plt.title('Stage 3 Features')
        plt.axis('off')
    
    if 'stage4' in feature_maps:
        feat_map = feature_maps['stage4'][0].mean(dim=0).cpu().numpy()
        feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min() + 1e-8)
        
        plt.subplot(2, 3, 5)
        plt.imshow(feat_map, cmap='viridis')
        plt.title('Stage 4 Features')
        plt.axis('off')
    
    # Final attention map
    if 'final' in attention_maps:
        att_map = attention_maps['final'][0, 0].cpu().numpy()
        att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)
        
        plt.subplot(2, 3, 6)
        plt.imshow(att_map, cmap='jet')
        plt.title('Final EigenCAM Attention')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'visualizations/epoch_{epoch}_attention.png')
    plt.close()

# Extract EigenCAM activation maps for visualization
def extract_eigencam_maps(model, image_tensor):
    model.eval()
    
    # Process the image and get feature maps and attention maps
    with torch.no_grad():
        outputs, feature_maps, attention_maps = model(image_tensor, return_attention=True)
    
    # Get class prediction
    _, predicted_class = outputs.max(1)
    
    return attention_maps, feature_maps, predicted_class