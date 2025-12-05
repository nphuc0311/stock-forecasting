"""
Training script for all forecasting models.

This script trains all enabled models on the provided dataset,
evaluates them, and saves the best model.
"""

import argparse
import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

from src.utils import (
    setup_logging, set_seed, load_config, get_device,
    apply_savgol_smoothing, inverse_log_transform
)
from src.dataset import (
    load_and_preprocess_data, split_data, create_dataloaders,
    prepare_inference_data
)
from src.models import get_model
from src.train import (
    train_all_models, select_best_model, predict, save_model
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train stock price forecasting models'
    )
    
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to training CSV file'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--save_model',
        action='store_true',
        default=True,
        help='Save the best model'
    )
    
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    return parser.parse_args()


def main():
    """Main training pipeline."""
    
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, 'training.log')
    setup_logging(log_level=args.log_level, log_file=log_file)
    
    logger.info("="*60)
    logger.info("Stock Price Forecasting - Training Pipeline")
    logger.info("="*60)
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Configuration loaded from {args.config}")
    
    # Set seed for reproducibility
    set_seed(config['seed'])
    
    # Get device
    device = get_device(config['training']['device'])
    
    # Load and preprocess data
    logger.info(f"Loading data from {args.data_path}")
    df = load_and_preprocess_data(args.data_path)
    
    # Split data
    df_train, df_val, df_test = split_data(
        df,
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio']
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        df_train, df_val, df_test,
        input_len=config['data']['input_len'],
        output_len=config['data']['output_len'],
        batch_size=config['training']['batch_size'],
        target_col=config['data']['target_col']
    )
    
    # Initialize models
    logger.info("\nInitializing models...")
    model_configs = {}
    
    for model_key, model_info in config['models'].items():
        if model_info.get('enabled', True):
            model_name = model_info['name']
            try:
                model = get_model(model_name, config)
                model_configs[model_name] = model
                logger.info(f"✓ {model_name} initialized")
            except Exception as e:
                logger.error(f"✗ Failed to initialize {model_name}: {e}")
    
    if not model_configs:
        logger.error("No models to train!")
        sys.exit(1)
    
    # Train all models
    logger.info(f"\nTraining {len(model_configs)} models...")
    trained_models = train_all_models(
        model_configs,
        train_loader,
        val_loader,
        test_loader,
        config,
        device
    )
    
    # Select best model
    best_model_name, best_model = select_best_model(
        trained_models,
        criterion='val_loss'
    )
    
    # Generate predictions with best model
    logger.info("\nGenerating predictions with best model...")
    input_data = prepare_inference_data(
        df,
        config['data']['input_len'],
        config['data']['target_col']
    )
    
    predictions_log = predict(best_model, input_data, device)
    predictions_log = predictions_log.flatten()
    
    # Convert to price
    predictions_price = inverse_log_transform(predictions_log)
    
    # Apply smoothing if enabled
    if config['postprocessing']['smoothing']['enabled']:
        window_length = min(
            config['postprocessing']['smoothing']['window_length'],
            len(predictions_price) if len(predictions_price) % 2 == 1 else len(predictions_price) - 1
        )
        predictions_price_smooth = apply_savgol_smoothing(
            predictions_price,
            window_length=window_length,
            polyorder=config['postprocessing']['smoothing']['polyorder']
        )
    else:
        predictions_price_smooth = predictions_price
    
    # Save predictions
    submission_df = pd.DataFrame({
        'id': range(1, len(predictions_price_smooth) + 1),
        'close': predictions_price_smooth
    })
    
    submission_path = os.path.join(args.output_dir, 'submission.csv')
    submission_df.to_csv(submission_path, index=False)
    logger.info(f"Predictions saved to {submission_path}")
    
    # Save training results
    results_summary = []
    for name, info in trained_models.items():
        results_summary.append({
            'Model': name,
            'Train_Loss': info['train_losses'][-1],
            'Val_Loss': info['best_val_loss'],
            'Test_Loss': info['test_loss']
        })
    
    results_df = pd.DataFrame(results_summary)
    results_path = os.path.join(args.output_dir, 'training_results.csv')
    results_df.to_csv(results_path, index=False)
    logger.info(f"Training results saved to {results_path}")
    
    # Save all trained models if requested
    if args.save_model:
        models_dir = os.path.join(args.output_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Save the best version of each model
        for model_name, model_info in trained_models.items():
            model_path = os.path.join(models_dir, f'{model_name}_best.pth')
            save_model(model_info['model'], model_path, model_name, config)
            logger.info(f"Saved best version of {model_name} to {model_path}")
    
    # Print final summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETED")
    logger.info("="*60)
    logger.info(f"Best Model: {best_model_name}")
    logger.info(f"Val Loss: {trained_models[best_model_name]['best_val_loss']:.6f}")
    logger.info(f"Test Loss: {trained_models[best_model_name]['test_loss']:.6f}")
    logger.info(f"Predictions: {len(predictions_price_smooth)} days")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()