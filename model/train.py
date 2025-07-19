import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from model import TaikoModel
from taiko_dataset import TaikoDataset, pad_collate
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths to your data
audio_root = r"D:\taiko_ai\taiko-autochart\mel_features"
label_root = r"D:\taiko_ai\taiko-autochart\dataset-labels-pt"

# Dataset
full_dataset = TaikoDataset(audio_root=audio_root, label_root=label_root)
print(f"Total samples: {len(full_dataset)}")

# Split ratios
train_ratio = 0.7  # 70% for training
val_ratio = 0.2    # 20% for validation
test_ratio = 0.1   # 10% for testing

# Calculate split sizes
total_size = len(full_dataset)
train_size = int(train_ratio * total_size)
val_size = int(val_ratio * total_size)
test_size = total_size - train_size - val_size  # Remainder goes to test

print(f"Train size: {train_size}")
print(f"Validation size: {val_size}")
print(f"Test size: {test_size}")

# Create random splits
torch.manual_seed(42)  # For reproducible splits
train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, 
    [train_size, val_size, test_size]
)

# Create DataLoaders
batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)

# Check label dimensions for model setup
print("Checking label dimensions...")
sample_audio, sample_label = full_dataset[0]
print(f"Sample label shape: {sample_label.shape}")

# Find maximum feature dimension
max_features = 0
for i in range(min(20, len(full_dataset))):  # Check first 20 samples
    _, label = full_dataset[i]
    max_features = max(max_features, label.shape[1])
print(f"Maximum features found: {max_features}")

# Initialize model
model = TaikoModel(output_dim=max_features)
model.to(device)

# Loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training parameters
num_epochs = 50
best_val_loss = float('inf')
patience = 10
patience_counter = 0

# Create directory for model checkpoints
os.makedirs('checkpoints', exist_ok=True)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for audio_batch, label_batch in loader:
        optimizer.zero_grad()
        
        audio_batch = audio_batch.to(device)
        label_batch = label_batch.float().to(device)
        
        # Forward pass
        preds = model(audio_batch)
        
        # Align dimensions
        target_seq_len = label_batch.size(1)
        if preds.size(1) != target_seq_len:
            if preds.size(1) > target_seq_len:
                preds = preds[:, :target_seq_len, :]
            else:
                pad_len = target_seq_len - preds.size(1)
                preds = F.pad(preds, (0, 0, 0, pad_len))
        
        target_feat_len = label_batch.size(2)
        if preds.size(2) != target_feat_len:
            if preds.size(2) > target_feat_len:
                preds = preds[:, :, :target_feat_len]
            else:
                pad_len = target_feat_len - preds.size(2)
                preds = F.pad(preds, (0, pad_len))
        
        # Loss and backprop
        loss = criterion(preds, label_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for audio_batch, label_batch in loader:
            audio_batch = audio_batch.to(device)
            label_batch = label_batch.float().to(device)
            
            # Forward pass
            preds = model(audio_batch)
            
            # Align dimensions (same as training)
            target_seq_len = label_batch.size(1)
            if preds.size(1) != target_seq_len:
                if preds.size(1) > target_seq_len:
                    preds = preds[:, :target_seq_len, :]
                else:
                    pad_len = target_seq_len - preds.size(1)
                    preds = F.pad(preds, (0, 0, 0, pad_len))
            
            target_feat_len = label_batch.size(2)
            if preds.size(2) != target_feat_len:
                if preds.size(2) > target_feat_len:
                    preds = preds[:, :, :target_feat_len]
                else:
                    pad_len = target_feat_len - preds.size(2)
                    preds = F.pad(preds, (0, pad_len))
            
            loss = criterion(preds, label_batch)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

# Training loop with validation
print("\nStarting training...")
for epoch in range(num_epochs):
    # Train
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # Validate
    val_loss = validate_epoch(model, val_loader, criterion, device)
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {train_loss:.6f}")
    print(f"  Val Loss: {val_loss:.6f}")
    
    # Early stopping and model saving
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, 'checkpoints/best_model.pth')
        print(f"  ✅ New best model saved! (Val Loss: {val_loss:.6f})")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\n⏹️  Early stopping triggered after {patience} epochs without improvement")
            break
    
    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, f'checkpoints/checkpoint_epoch_{epoch+1}.pth')
        print(f"  💾 Checkpoint saved at epoch {epoch+1}")

# Final evaluation on test set
print("\n🧪 Evaluating on test set...")
test_loss = validate_epoch(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.6f}")

# Load best model for final save
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
torch.save(model.state_dict(), 'taiko_model_final.pth')
print("\n✅ Training completed!")
print(f"📊 Best validation loss: {best_val_loss:.6f}")
print(f"📊 Final test loss: {test_loss:.6f}")
print("📁 Final model saved as 'taiko_model_final.pth'")