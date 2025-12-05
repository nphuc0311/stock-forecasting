import random
import logging
from typing import Optional, List
import numpy as np
import torch
import yaml
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"✓ Random seeds fixed at {seed}")


def load_config(config_path: str) -> dict:
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise


def apply_savgol_smoothing(
    predictions: np.ndarray,
    window_length: int = 9,
    polyorder: int = 2
) -> np.ndarray:
    if window_length % 2 == 0:
        logger.warning(f"window_length must be odd, adjusting {window_length} to {window_length + 1}")
        window_length += 1
    
    if window_length > len(predictions):
        logger.warning(
            f"window_length ({window_length}) > prediction length ({len(predictions)}), "
            f"adjusting to {len(predictions) if len(predictions) % 2 == 1 else len(predictions) - 1}"
        )
        window_length = len(predictions) if len(predictions) % 2 == 1 else len(predictions) - 1
    
    smoothed = savgol_filter(predictions, window_length=window_length, polyorder=polyorder)
    
    logger.info(f"Applied Savitzky-Golay smoothing (window={window_length}, poly={polyorder})")
    logger.debug(f"  Before smoothing (first 5): {predictions[:5]}")
    logger.debug(f"  After smoothing (first 5):  {smoothed[:5]}")
    
    return smoothed


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None
) -> None:
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    logger.info("Logging configured successfully")


def get_device(device_name: str = "cuda") -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        logger.info("Using device: CPU")
    
    return device


def inverse_log_transform(log_values: np.ndarray) -> np.ndarray:
    return np.exp(log_values)


def calculate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray
) -> dict:
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))
    mape = np.mean(np.abs((predictions - targets) / targets)) * 100
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }


def format_model_summary(
    model_name: str,
    train_loss: float,
    val_loss: float,
    test_loss: float
) -> str:
    return (
        f"\n{'='*60}\n"
        f"Model: {model_name}\n"
        f"{'='*60}\n"
        f"Train Loss: {train_loss:.6f}\n"
        f"Val Loss:   {val_loss:.6f}\n"
        f"Test Loss:  {test_loss:.6f}\n"
        f"{'='*60}\n"
    )


class EarlyStopping:    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        verbose: bool = True
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(
        self,
        val_loss: float,
        model: torch.nn.Module
    ) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"EarlyStopping counter: {self.counter}/{self.patience} "
                    f"(best: {self.best_loss:.6f})"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
        
        return self.early_stop
    
    def save_checkpoint(self, model: torch.nn.Module) -> None:
        import copy
        self.best_model_state = copy.deepcopy(model.state_dict())
        if self.verbose:
            logger.debug(f"Model checkpoint saved (loss: {self.best_loss:.6f})")
    
    def load_best_model(self, model: torch.nn.Module) -> None:
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            logger.info(f"Loaded best model (val loss: {self.best_loss:.6f})")