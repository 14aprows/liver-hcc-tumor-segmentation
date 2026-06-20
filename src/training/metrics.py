import torch
def dice_score(
    logits,
    targets,
    threshold,
    smooth
):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    
    preds = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice

def iou_score(
    logits,
    targets,
    threshold,
    smooth
):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    
    preds = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou