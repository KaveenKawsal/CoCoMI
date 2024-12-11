import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Import necessary classes from the original training script
from training2 import (
    AttentionUNet, 
    SegmentationDataset, 
    setup_device, 
    setup_data_loader,
    create_datasets
)

def load_model_with_history(model_path, num_classes=5):
    """Load a saved model and its training history."""
    device = setup_device()
    model = AttentionUNet(n_channels=3, n_classes=num_classes).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Extract training history from checkpoint
    training_history = {
        'train_losses': checkpoint.get('loss', []),  # Assuming losses are stored in checkpoint
        'val_losses': checkpoint.get('val_loss', []),
        'train_accuracies': checkpoint.get('train_acc', []),
        'val_accuracies': checkpoint.get('val_acc', []),
        'train_f1s': checkpoint.get('train_f1', []),
        'val_f1s': checkpoint.get('val_f1', []),
        'epochs': checkpoint.get('epoch', 0)
    }
    
    model.eval()
    return model, device, training_history

def evaluate_model_full(model, test_loader, device, class_names):
    """Comprehensive model evaluation."""
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_targets.extend(masks.cpu().numpy().flatten())
            all_probabilities.extend(probabilities.cpu().numpy().reshape(-1, probabilities.shape[1]))

    # Compute metrics
    cm = confusion_matrix(all_targets, all_predictions)
    report = classification_report(all_targets, all_predictions, target_names=class_names)
    
    # Compute ROC and AUC for each class
    all_probabilities = np.array(all_probabilities)
    all_targets = np.array(all_targets)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve((all_targets == i).astype(int), all_probabilities[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return cm, report, fpr, tpr, roc_auc

def plot_confusion_matrix(cm, class_names):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.2%})',
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('figure_3_confusion_matrix.png', bbox_inches='tight')
    plt.close()

def plot_roc_curves(fpr, tpr, roc_auc, class_names):
    """Plot ROC curves for each class."""
    plt.figure(figsize=(10, 8))
    for i in range(len(class_names)):
        plt.plot(fpr[i], tpr[i], 
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curves.png')
    plt.close()

def plot_training_metrics(training_history):
    """Plot training and validation metrics."""
    plt.figure(figsize=(15, 10))
    
    # Accuracy plot
    plt.subplot(2, 2, 1)
    plt.plot(training_history['train_accuracies'], label='Training')
    plt.plot(training_history['val_accuracies'], label='Validation')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss plot
    plt.subplot(2, 2, 2)
    plt.plot(training_history['train_losses'], label='Training')
    plt.plot(training_history['val_losses'], label='Validation')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # F1 Score plot
    plt.subplot(2, 2, 3)
    plt.plot(training_history['train_f1s'], label='Training')
    plt.plot(training_history['val_f1s'], label='Validation')
    plt.title('F1 Score over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

def plot_epoch_metrics(training_history):
    """Plot metrics at different epoch checkpoints."""
    epochs = list(range(5, training_history['epochs'] + 1, 5))
    
    # Get metrics at checkpoints
    checkpoint_accuracies = {
        e: training_history['val_accuracies'][e-1] 
        for e in epochs if e <= len(training_history['val_accuracies'])
    }
    
    checkpoint_f1s = {
        e: training_history['val_f1s'][e-1] 
        for e in epochs if e <= len(training_history['val_f1s'])
    }
    
    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.bar(checkpoint_accuracies.keys(), checkpoint_accuracies.values())
    plt.title('Validation Accuracy at Different Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('epoch_accuracies.png')
    plt.close()
    
    # Plot F1 scores
    plt.figure(figsize=(10, 5))
    plt.bar(checkpoint_f1s.keys(), checkpoint_f1s.values())
    plt.title('Validation F1 Scores at Different Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.savefig('epoch_f1_scores.png')
    plt.close()

def plot_accuracy_bar_graph(epochs_accuracy):
    """
    Plot bar graph of testing accuracy for different epochs.
    
    Args:
    epochs_accuracy (dict): Dictionary of epoch numbers and their accuracies
    """
    plt.figure(figsize=(10, 5))
    plt.bar(epochs_accuracy.keys(), epochs_accuracy.values())
    plt.title('Testing Accuracy for Different Epochs')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy')
    plt.savefig('figure_1_testing_accuracy.png')
    plt.close()

def plot_f1_score_bar_graph(epochs_f1_scores):
    """
    Plot bar graph of F1 scores for different epochs.
    
    Args:
    epochs_f1_scores (dict): Dictionary of epoch numbers and their F1 scores
    """
    plt.figure(figsize=(10, 5))
    plt.bar(epochs_f1_scores.keys(), epochs_f1_scores.values())
    plt.title('F1 Scores for Different Epochs')
    plt.xlabel('Number of Epochs')
    plt.ylabel('F1 Score')
    plt.savefig('figure_2_f1_scores.png')
    plt.close()

def main():
    # Configuration
    ROOT_DIR = r"S:\SEM3\COMPUTER NETWORKS\project\train"
    MODEL_PATH = os.path.join(os.getcwd(), 'saved_models', 'final_unet_model.pth')
    BATCH_SIZE = 2

    # Create datasets
    train_dataset, val_dataset, class_names = create_datasets(ROOT_DIR)
    test_loader = setup_data_loader(val_dataset, BATCH_SIZE)

    # Load model and training history
    model, device, training_history = load_model_with_history(MODEL_PATH)

    # Evaluate model
    cm, report, fpr, tpr, roc_auc = evaluate_model_full(model, test_loader, device, class_names)

    # Print classification report
    print("Classification Report:")
    print(report)

    # Generate plots
    plot_confusion_matrix(cm, class_names)
    plot_roc_curves(fpr, tpr, roc_auc, class_names)
    plot_training_metrics(training_history)
    plot_epoch_metrics(training_history)

    # Save detailed metrics
    with open('model_evaluation_metrics.txt', 'w') as f:
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\nROC AUC Scores:\n")
        for i, name in enumerate(class_names):
            f.write(f"{name}: {roc_auc[i]:.4f}\n")
        
        f.write("\n\nTraining History Summary:\n")
        f.write(f"Final Training Loss: {training_history['train_losses'][-1]:.4f}\n")
        f.write(f"Final Validation Loss: {training_history['val_losses'][-1]:.4f}\n")
        f.write(f"Final Training Accuracy: {training_history['train_accuracies'][-1]:.4f}\n")
        f.write(f"Final Validation Accuracy: {training_history['val_accuracies'][-1]:.4f}\n")
        f.write(f"Final Training F1: {training_history['train_f1s'][-1]:.4f}\n")
        f.write(f"Final Validation F1: {training_history['val_f1s'][-1]:.4f}\n")

if __name__ == "__main__":
    main()