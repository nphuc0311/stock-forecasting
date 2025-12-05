from typing import Tuple, Optional
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        input_len: int,
        output_len: int,
        target_col: str = 'close_log'
    ) -> None:
        self.data = data[target_col].values
        self.input_len = input_len
        self.output_len = output_len
        
        logger.info(
            f"Dataset initialized with {len(self)} samples "
            f"(input_len={input_len}, output_len={output_len})"
        )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data) - self.input_len - self.output_len + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx:idx + self.input_len]
        y = self.data[idx + self.input_len:idx + self.input_len + self.output_len]
        return torch.FloatTensor(x), torch.FloatTensor(y)


def load_and_preprocess_data(
    file_path: str,
    time_col: str = 'time',
    target_col: str = 'close'
) -> pd.DataFrame:
    logger.info(f"Loading data from {file_path}")
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    
    # Validate columns
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in data")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    # Preprocess
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)
    
    # Log transformation
    df[f'{target_col}_log'] = np.log(df[target_col])
    
    logger.info(
        f"Data loaded: {len(df)} rows, "
        f"date range: {df[time_col].min()} to {df[time_col].max()}"
    )
    
    return df


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_len = len(df)
    train_end = int(total_len * train_ratio)
    val_end = int(total_len * (train_ratio + val_ratio))

    df_train = df.iloc[:train_end].reset_index(drop=True)
    df_val = df.iloc[train_end:val_end].reset_index(drop=True)
    df_test = df.iloc[val_end:].reset_index(drop=True)

    logger.info("Data split completed:")
    logger.info(f"  Train: {len(df_train)} samples ({train_ratio*100:.1f}%)")
    logger.info(f"  Val:   {len(df_val)} samples ({val_ratio*100:.1f}%)")
    logger.info(f"  Test:  {len(df_test)} samples ({(1-train_ratio-val_ratio)*100:.1f}%)")

    return df_train, df_val, df_test


def create_dataloaders(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    input_len: int,
    output_len: int,
    batch_size: int,
    target_col: str = 'close_log'
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    # Training dataset
    train_dataset = TimeSeriesDataset(df_train, input_len, output_len, target_col)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Validation dataset (need historical context)
    df_for_val = pd.concat([df_train, df_val], ignore_index=True)
    val_dataset = TimeSeriesDataset(df_for_val, input_len, output_len, target_col)
    val_start_idx = len(df_train) - input_len
    val_dataset_filtered = [val_dataset[i] for i in range(len(val_dataset)) if i >= val_start_idx]
    val_loader = DataLoader(val_dataset_filtered, batch_size=batch_size, shuffle=False)
    
    # Test dataset (need historical context)
    df_for_test = pd.concat([df_train, df_val, df_test], ignore_index=True)
    test_dataset = TimeSeriesDataset(df_for_test, input_len, output_len, target_col)
    test_start_idx = len(df_train) + len(df_val) - input_len
    test_dataset_filtered = [test_dataset[i] for i in range(len(test_dataset)) if i >= test_start_idx]
    test_loader = DataLoader(test_dataset_filtered, batch_size=batch_size, shuffle=False)
    
    logger.info("DataLoaders created:")
    logger.info(f"  Train: {len(train_loader)} batches")
    logger.info(f"  Val:   {len(val_loader)} batches")
    logger.info(f"  Test:  {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


def prepare_inference_data(
    df: pd.DataFrame,
    input_len: int,
    target_col: str = 'close_log'
) -> torch.Tensor:

    # Take the last input_len values
    data = df[target_col].values[-input_len:]
    
    if len(data) < input_len:
        raise ValueError(
            f"Insufficient data for inference. Need {input_len} points, got {len(data)}"
        )
    
    return torch.FloatTensor(data).unsqueeze(0)  # Add batch dimension