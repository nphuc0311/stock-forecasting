import logging
from typing import Dict, List, Tuple, Optional
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils import EarlyStopping

logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:

    model.train()
    epoch_loss = 0.0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    return avg_loss


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, List[np.ndarray], List[np.ndarray]]:

    model.eval()
    epoch_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            epoch_loss += loss.item()
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    avg_loss = epoch_loss / len(data_loader) if len(data_loader) > 0 else float('inf')
    return avg_loss, all_predictions, all_targets


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str,
    config: dict,
    device: torch.device
) -> Tuple[nn.Module, List[float], List[float], float]:

    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_name}")
    logger.info(f"{'='*60}")
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )
    
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping_patience'],
        verbose=True
    )
    
    train_losses = []
    val_losses = []
    num_epochs = config['training']['num_epochs']
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"Epoch [{epoch+1}/{num_epochs}] - "
                f"Train Loss: {train_loss:.6f}, "
                f"Val Loss: {val_loss:.6f}"
            )
        
        if early_stopping(val_loss, model):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    early_stopping.load_best_model(model)
    best_val_loss = early_stopping.best_loss
    
    logger.info(f"Training completed! Best validation loss: {best_val_loss:.6f}")
    
    return model, train_losses, val_losses, best_val_loss


def train_all_models(
    model_configs: Dict[str, nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: dict,
    device: torch.device
) -> Dict[str, dict]:

    trained_models = {}
    
    for model_name, model in model_configs.items():
        trained_model, train_losses, val_losses, best_val_loss = train_model(
            model, train_loader, val_loader, model_name, config, device
        )
        
        criterion = nn.MSELoss()
        test_loss, predictions, targets = evaluate(
            trained_model, test_loader, criterion, device
        )
        
        logger.info(f"{model_name} - Test Loss: {test_loss:.6f}")
        
        trained_models[model_name] = {
            'model': trained_model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'test_loss': test_loss,
            'predictions': predictions,
            'targets': targets
        }
    
    return trained_models


def select_best_model(
    trained_models: Dict[str, dict],
    criterion: str = 'val_loss'
) -> Tuple[str, nn.Module]:
    
    if criterion == 'val_loss':
        best_model_name = min(
            trained_models,
            key=lambda k: trained_models[k]['best_val_loss']
        )
    elif criterion == 'test_loss':
        best_model_name = min(
            trained_models,
            key=lambda k: trained_models[k]['test_loss']
        )
    else:
        raise ValueError(f"Unknown criterion: {criterion}")
    
    best_model = trained_models[best_model_name]['model']
    
    logger.info(f"\n{'='*60}")
    logger.info(f"BEST MODEL SELECTION (criterion: {criterion})")
    logger.info(f"{'='*60}")
    
    for name, info in trained_models.items():
        marker = ">>>" if name == best_model_name else "   "
        logger.info(
            f"{marker} {name:15s} - "
            f"Val Loss: {info['best_val_loss']:.6f}, "
            f"Test Loss: {info['test_loss']:.6f}"
        )
    
    logger.info(f"\nSelected: {best_model_name}")
    logger.info(f"{'='*60}")
    
    return best_model_name, best_model


def predict(
    model: nn.Module,
    input_data: torch.Tensor,
    device: torch.device
) -> np.ndarray:

    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        input_data = input_data.to(device)
        output = model(input_data)
        predictions = output.cpu().numpy()
    
    return predictions


def save_model(
    model: nn.Module,
    save_path: str,
    model_name: str,
    config: dict
) -> None:

    checkpoint = {
        'model_name': model_name,
        'model_state_dict': model.state_dict(),
        'config': config
    }
    
    torch.save(checkpoint, save_path)
    logger.info(f"Model saved to {save_path}")


def load_model(
    load_path: str,
    device: torch.device
) -> Tuple[nn.Module, str, dict]:

    checkpoint = torch.load(load_path, map_location=device)
    
    model_name = checkpoint['model_name']
    config = checkpoint['config']
    
    from src.models import get_model
    model = get_model(model_name, config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    logger.info(f"Model loaded from {load_path}")
    
    return model, model_name, config