# main.py

import os
import argparse
import logging
import torch

# Import from our new modules
from utils import (
    set_seed, setup_logging, get_device, get_optimizer, get_scheduler,
    calculate_class_weights, setup_model_and_loss, save_model_checkpoint
)
from dataset import create_datasets, setup_data_loader
from engine import train_model
from visualize import plot_experiment_comparison, plot_final_metrics_bar_chart

# Disable Albumentations update check
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

def run_ablation_study(args, device, train_dataset, val_dataset, class_names):
    """Performs an ablation study with different models and loss functions."""
    logging.info("Starting Ablation Study...")
    
    variants = [
        {'model_type': 'vanilla_unet', 'loss_type': 'cross_entropy', 'name': 'Vanilla_UNet_CrossEntropy'},
        {'model_type': 'vanilla_unet', 'loss_type': 'dice', 'name': 'Vanilla_UNet_Dice'},
        {'model_type': 'vanilla_unet', 'loss_type': 'focal', 'name': 'Vanilla_UNet_Focal'},
        {'model_type': 'attention_unet', 'loss_type': 'cross_entropy', 'name': 'Attention_UNet_CrossEntropy'},
        {'model_type': 'attention_unet', 'loss_type': 'dice', 'name': 'Attention_UNet_Dice'},
        {'model_type': 'attention_unet', 'loss_type': 'focal', 'name': 'Attention_UNet_Focal'},
        {'model_type': 'attention_unet', 'loss_type': 'advanced_focal_dice', 'name': 'Attention_UNet_AdvancedFocalDice'},
    ]
    
    results = {}
    
    for variant in variants:
        logging.info(f"\n{'='*60}\nTraining Variant: {variant['name']}\n{'='*60}")
        
        # Data loaders
        train_loader = setup_data_loader(train_dataset, args.batch_size, shuffle=True)
        val_loader = setup_data_loader(val_dataset, args.batch_size, shuffle=False)
        
        # Calculate weights if needed
        class_weights = None
        if variant['loss_type'] in ['advanced_focal_dice', 'cross_entropy']:
            logging.info("Calculating class weights...")
            class_weights = calculate_class_weights(train_loader, len(class_names), device)
            
        # Setup model, loss, optimizer, scheduler
        model, criterion = setup_model_and_loss(
            variant['model_type'], variant['loss_type'], len(class_names), device, class_weights
        )
        optimizer = get_optimizer('adam', model.parameters(), args.learning_rate)
        scheduler = get_scheduler(args.scheduler, optimizer, T_max=args.num_epochs)
        
        # Train model
        model, metrics = train_model(
            model, train_loader, val_loader, criterion, optimizer, 
            scheduler, device, args.num_epochs, class_names
        )
        
        # Store results
        results[variant['name']] = {
            'final_train_loss': metrics['train_losses'][-1],
            'final_val_loss': metrics['val_losses'][-1],
            'final_val_accuracy': metrics['val_accuracies'][-1],
            'final_val_f1': metrics['val_f1s'][-1],
            'final_val_jaccard': metrics['val_jaccards'][-1],
            'val_losses': metrics['val_losses'],
            'val_accuracies': metrics['val_accuracies'],
            'val_f1s': metrics['val_f1s'],
            'val_jaccards': metrics['val_jaccards']
        }
        
        # Save model
        save_path = os.path.join(args.save_dir, 'ablation', f'{variant["name"]}.pth')
        save_model_checkpoint(model, optimizer, args.num_epochs, metrics, save_path)
        
        del model, criterion, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()
    
    # Plotting results
    ablation_save_dir = os.path.join(args.save_dir, 'ablation_plots')
    os.makedirs(ablation_save_dir, exist_ok=True)
    
    plot_experiment_comparison(results, 'val_losses', 'Ablation Study: Validation Loss', os.path.join(ablation_save_dir, 'val_loss_comparison.png'))
    plot_experiment_comparison(results, 'val_f1s', 'Ablation Study: Validation F1 Score', os.path.join(ablation_save_dir, 'val_f1_comparison.png'))
    plot_final_metrics_bar_chart(results, os.path.join(ablation_save_dir, 'final_metrics_comparison.png'))

    logging.info(f"Ablation study complete. Results and plots saved in {args.save_dir}")

def main():
    parser = argparse.ArgumentParser(description='Coconut Tree Disease Segmentation')
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory of the dataset')
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save models and results')
    parser.add_argument('--ablation', action='store_true', help='Run a full ablation study')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training")
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['step', 'plateau', 'cosine'], help='LR scheduler')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    # Initial setup
    setup_logging()
    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.save_dir, exist_ok=True)

    # Data loading
    try:
        train_dataset, val_dataset, class_names = create_datasets(args.data_dir)
    except (ValueError, FileNotFoundError) as e:
        logging.error(f"Failed to create datasets: {e}")
        return

    if args.ablation:
        run_ablation_study(args, device, train_dataset, val_dataset, class_names)
    else:
        logging.error("No training mode specified. Use --ablation to run the study.")
        logging.info("To add a single training run, please extend main.py.")
        # NOTE: The original `--train` functionality is now part of the ablation study.
        # You could easily add a new block here for a single, specific training run if needed.

if __name__ == '__main__':
    main()