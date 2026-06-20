import torch
from tqdm import tqdm
from src.training.metrics import dice_score, iou_score

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    for images, masks in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            total_dice += dice_score(logits, masks).item()
            total_iou += iou_score(logits, masks).item()

    num_batches = len(dataloader)

    if num_batches == 0:
        raise RuntimeError("Training dataloader is empty.")

    return {
        "loss": total_loss / num_batches,
        "dice": total_dice / num_batches,
        "iou": total_iou / num_batches,
    }

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    for images, masks in tqdm(dataloader, desc="Evaluating", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        total_loss += loss.item()
        total_dice += dice_score(logits, masks).item()
        total_iou += iou_score(logits, masks).item()

    num_batches = len(dataloader)

    if num_batches == 0:
        raise RuntimeError("Evaluation dataloader is empty.")

    return {
        "loss": total_loss / num_batches,
        "dice": total_dice / num_batches,
        "iou": total_iou / num_batches,
    }