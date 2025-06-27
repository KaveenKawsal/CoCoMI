# engine.py

import torch
import torch.nn.functional as F
import logging
import copy
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, classification_report

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Runs a single training epoch."""
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for images, masks in progress_bar:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(masks.cpu().numpy().flatten())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    jaccard = jaccard_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, accuracy, f1, jaccard

def evaluate_model(model, dataloader, criterion, device, class_names):
    """Evaluates the model on a dataset."""
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
        for images, masks in progress_bar:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(masks.cpu().numpy().flatten())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    jaccard = jaccard_score(all_targets, all_preds, average='weighted', zero_division=0)
    report = classification_report(
        all_targets, all_preds, 
        target_names=class_names, 
        output_dict=True,
        zero_division=0
    )

    return avg_loss, accuracy, f1, jaccard, report

def train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler, device, 
    num_epochs, class_names, patience=15
):
    """The main training loop with early stopping."""
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    metrics = {
        'train_losses': [], 'val_losses': [],
        'train_accuracies': [], 'val_accuracies': [],
        'train_f1s': [], 'val_f1s': [],
        'train_jaccards': [], 'val_jaccards': [],
        'val_reports': []
    }

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc, train_f1, train_jaccard = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1, val_jaccard, val_report = evaluate_model(
            model, val_loader, criterion, device, class_names
        )
        
        # Store metrics
        metrics['train_losses'].append(train_loss)
        metrics['val_losses'].append(val_loss)
        metrics['train_accuracies'].append(train_acc)
        metrics['val_accuracies'].append(val_acc)
        metrics['train_f1s'].append(train_f1)
        metrics['val_f1s'].append(val_f1)
        metrics['train_jaccards'].append(train_jaccard)
        metrics['val_jaccards'].append(val_jaccard)
        metrics['val_reports'].append(val_report)

        logging.info(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}"
        )

        # Handle scheduler step based on its type
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Early stopping and best model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            logging.info(f"New best model found with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered at epoch {epoch}.")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, metrics