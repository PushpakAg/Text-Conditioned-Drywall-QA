import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import amp

import config
from dataset import CTDataset
from model import ModulatedUNet
from utils import save_checkpoint, dice_score

def train_fn(loader, model, optimizer, loss_fn, scaler, device):
    loop = tqdm(loader, desc="Training")
    total_loss = 0.0
    model.train()
    for batch_idx, batch in enumerate(loop):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        text_tokens = {key: val.to(device) for key, val in batch["text_tokens"].items()}
        with amp.autocast(device):
            predictions = model(images, text_tokens)
            loss = loss_fn(predictions, masks)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    avg_loss = total_loss / len(loader)
    return avg_loss

def validate_fn(loader, model, loss_fn, device):
    loop = tqdm(loader, desc="Validating")
    total_loss = 0.0
    total_dice = 0.0
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loop):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            text_tokens = {key: val.to(device) for key, val in batch["text_tokens"].items()}
            predictions = model(images, text_tokens)
            loss = loss_fn(predictions, masks)
            dice = dice_score(predictions, masks)
            total_loss += loss.item()
            total_dice += dice
            loop.set_postfix(val_loss=loss.item(), val_dice=dice)
    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)
    model.train()
    return avg_loss, avg_dice

def main():
    print("Setting up DataLoaders...")
    train_dataset = CTDataset(
        data_dir=config.DATA_DIR,
        split='train',
        text_model_name=config.TEXT_MODEL_NAME,
        image_size=config.IMAGE_SIZE
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        shuffle=True
    )
    valid_dataset = CTDataset(
        data_dir=config.DATA_DIR,
        split='valid',
        text_model_name=config.TEXT_MODEL_NAME,
        image_size=config.IMAGE_SIZE
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        shuffle=False
    )
    print(f"Initializing model on device: {config.DEVICE}")
    model = ModulatedUNet(
        text_model_name=config.TEXT_MODEL_NAME,
        freeze_encoders=config.FREEZE_ENCODERS
    ).to(config.DEVICE)

    loss_fn_bce = nn.BCEWithLogitsLoss()    
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scaler = amp.GradScaler('cuda')    
    best_valid_dice = 0.0
    
    for epoch in range(config.EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{config.EPOCHS} ---")
        train_loss = train_fn(train_loader, model, optimizer, loss_fn_bce, scaler, config.DEVICE)
        valid_loss, valid_dice = validate_fn(valid_loader, model, loss_fn_bce, config.DEVICE)
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Valid Loss: {valid_loss:.4f}")
        print(f"  Valid Dice Score: {valid_dice:.4f}")

        if valid_dice > best_valid_dice:
            print(f"Dice score improved from {best_valid_dice:.4f} to {valid_dice:.4f}")
            best_valid_dice = valid_dice
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_valid_dice": best_valid_dice,
            }
            save_checkpoint(checkpoint, directory=config.CHECKPOINT_DIR, filename="best_model.pth.tar")

    print("Training Complete!")

if __name__ == "__main__":
    main()