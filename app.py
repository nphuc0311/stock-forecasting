import os
import logging
import tempfile
from typing import Tuple
import pandas as pd
import numpy as np
import torch
import gradio as gr
import plotly.graph_objects as go

from src.utils import (
    setup_logging, set_seed, load_config, get_device,
    apply_savgol_smoothing, inverse_log_transform
)
from src.dataset import load_and_preprocess_data, prepare_inference_data
from src.models import get_model
from src.train import load_model, predict

# Setup logging
setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)

# Load configuration
CONFIG_PATH = "configs/config.yaml"
config = load_config(CONFIG_PATH)

# Set seed for reproducibility
set_seed(config['seed'])

# Get device
device = get_device(config['training']['device'])

# Available models
MODEL_OPTIONS = [
    "LinearModel",
    "NLinearModel",
    "DLinearModel",
    "xLSTMModel",
    "xLSTMTimeModel",
    "TimeMixerModel"
]

# Model paths
MODELS_DIR = "results/models"
SAVED_MODELS = {}

def discover_saved_models():
    global SAVED_MODELS
    SAVED_MODELS.clear()  # Clear previous cache
    if os.path.exists(MODELS_DIR):
        for file in os.listdir(MODELS_DIR):
            if file.endswith('_best.pth'):
                model_name = file.replace('_best.pth', '')
                model_path = os.path.join(MODELS_DIR, file)
                SAVED_MODELS[model_name] = model_path
                logger.info(f"Found saved model: {model_name} at {model_path}")
    if not SAVED_MODELS:
        logger.warning(f"No saved models found in {MODELS_DIR}")
    return SAVED_MODELS

# Discover saved models on startup
discover_saved_models()

def create_forecast_plot(
    historical_data: pd.DataFrame,
    predictions: np.ndarray,
    forecast_days: int,
    time_col: str = 'time',
    price_col: str = 'close'
) -> go.Figure:
    # Ensure time column is datetime
    historical_data = historical_data.copy()
    historical_data[time_col] = pd.to_datetime(historical_data[time_col])

    fig = go.Figure()

    # Historical data trace
    fig.add_trace(go.Scatter(
        x=historical_data[time_col],
        y=historical_data[price_col],
        mode='lines',
        name='Historical Data',
        line=dict(width=2),
        hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Price</b>: %{y:.2f}<extra></extra>'
    ))

    # Compute forecast dates (start the day after last historical date)
    last_date = historical_data[time_col].iloc[-1]
    forecast_start_date = last_date + pd.Timedelta(days=1)

    forecast_dates = pd.date_range(
        start=forecast_start_date,
        periods=forecast_days,
        freq='D'
    )

    # Predictions trace
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=predictions,
        mode='lines+markers',
        name='Forecast',
        line=dict(width=2),
        marker=dict(size=4),
        hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Forecast</b>: %{y:.2f}<extra></extra>'
    ))

    # Add vertical line using add_shape + add_annotation to avoid add_vline aggregation bug
    fig.add_shape(
        dict(
            type="line",
            x0=last_date,
            x1=last_date,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="gray", dash="dash")
        )
    )

    fig.add_annotation(
        dict(
            x=last_date,
            y=1.02,
            xref="x",
            yref="paper",
            showarrow=False,
            text="Forecast Start",
            align="center"
        )
    )

    fig.update_layout(
        title='Stock Price Forecast',
        xaxis_title='Date',
        yaxis_title='Price (VND)',
        hovermode='x unified',
        template='plotly_white',
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    return fig


def forecast_stock(
    csv_file: gr.File,
    model_name: str,
    forecast_days: int,
    apply_smoothing: bool
) -> Tuple[go.Figure, pd.DataFrame, str]:
    try:
        logger.info(f"Processing forecast request: model={model_name}, days={forecast_days}")
        
        # Validate inputs
        if csv_file is None:
            return None, None, "❌ Please upload a CSV file"
        
        if forecast_days <= 0 or forecast_days > 365:
            return None, None, "❌ Forecast days must be between 1 and 365"
        
        # Get the file path from csv_file object
        file_path = csv_file.name if hasattr(csv_file, 'name') else csv_file
        
        # Load and preprocess data
        df = load_and_preprocess_data(file_path)
        
        if len(df) < config['data']['input_len']:
            return None, None, f"❌ Insufficient data. Need at least {config['data']['input_len']} rows"
        
        # Update config with requested forecast days
        forecast_config = config.copy()
        forecast_config['data']['output_len'] = forecast_days
        forecast_config['data']['total_predict_days'] = forecast_days
        
        # Load saved model if available, otherwise create fresh model
        if model_name in SAVED_MODELS:
            model_path = SAVED_MODELS[model_name]
            logger.info(f"Loading saved model from {model_path}")
            try:
                model, loaded_model_name, loaded_config = load_model(model_path, device)
                logger.info(f"Model {model_name} loaded successfully from {model_path}")
                # Update config from saved model
                forecast_config = loaded_config.copy()
                forecast_config['data']['output_len'] = forecast_days
                forecast_config['data']['total_predict_days'] = forecast_days
            except Exception as e:
                logger.warning(f"Failed to load saved model: {e}. Creating fresh model instead.")
                model = get_model(model_name, forecast_config)
                model = model.to(device)
        else:
            logger.warning(f"No saved model found for {model_name}. Creating fresh untrained model.")
            logger.warning("For best results, please run 'python train_models.py --data_path <data_path> --save_model' first")
            model = get_model(model_name, forecast_config)
            model = model.to(device)
        
        # Prepare input data
        input_data = prepare_inference_data(
            df,
            config['data']['input_len'],
            config['data']['target_col']
        )
        
        # Make predictions
        predictions_log = predict(model, input_data, device)
        predictions_log = predictions_log.flatten()
        
        # Convert from log space to price
        predictions_price = inverse_log_transform(predictions_log)
        
        # Apply smoothing if requested
        if apply_smoothing and config['postprocessing']['smoothing']['enabled']:
            window_length = min(
                config['postprocessing']['smoothing']['window_length'],
                len(predictions_price) if len(predictions_price) % 2 == 1 else len(predictions_price) - 1
            )
            predictions_price = apply_savgol_smoothing(
                predictions_price,
                window_length=window_length,
                polyorder=config['postprocessing']['smoothing']['polyorder']
            )
            logger.info("Smoothing applied to predictions")
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'Day': range(1, forecast_days + 1),
            'Predicted_Price': predictions_price
        })
        
        # Create plot
        fig = create_forecast_plot(df, predictions_price, forecast_days)
        
        status_msg = f"✅ Forecast completed successfully! Generated {forecast_days} days of predictions using {model_name}."
        
        logger.info(status_msg)
        
        return fig, results_df, status_msg
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return None, None, error_msg


# Create Gradio interface
def create_interface() -> gr.Blocks:    
    with gr.Blocks(title="Stock Price Forecasting", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """
            # 📈 Stock Price Forecasting System
            
            Upload your historical stock data and select a model to generate future price predictions.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Configuration")
                
                csv_input = gr.File(
                    label="Upload CSV File",
                    file_types=[".csv"],
                    type="filepath"
                )
                
                model_dropdown = gr.Dropdown(
                    choices=MODEL_OPTIONS,
                    value="TimeMixerModel",
                    label="Select Model",
                    info="Choose the forecasting model to use"
                )
                
                forecast_days_slider = gr.Slider(
                    minimum=1,
                    maximum=365,
                    value=100,
                    step=1,
                    label="Forecast Days",
                    info="Number of days to predict"
                )
                
                smoothing_checkbox = gr.Checkbox(
                    value=True,
                    label="Apply Savitzky-Golay Smoothing",
                    info="Smooth predictions to reduce noise"
                )
                
                predict_btn = gr.Button("🚀 Generate Forecast", variant="primary", size="lg")
                
                status_output = gr.Textbox(
                    label="Status",
                    lines=2,
                    interactive=False
                )
            
            with gr.Column(scale=2):
                gr.Markdown("### Results")
                
                plot_output = gr.Plot(label="Forecast Chart")
                
                predictions_output = gr.Dataframe(
                    label="Predictions Table",
                    headers=["Day", "Predicted_Price"],
                    datatype=["number", "number"],
                    row_count=10
                )
                
                download_btn = gr.DownloadButton(
                    label="📥 Download Predictions CSV",
                    visible=False
                )
        
        gr.Markdown(
            """
            ---
            ### 📝 Instructions:
            1. **Upload CSV**: Your CSV must contain at least columns: `time` (date) and `close` (price)
            2. **Select Model**: Choose from 6 different forecasting architectures
            3. **Set Forecast Days**: Specify how many days ahead to predict (1-365)
            4. **Apply Smoothing** (Optional): Reduces noise in predictions
            5. **Generate Forecast**: Click the button to run the prediction
            
            ### 🎯 Models Available:
            - **LinearModel**: Simple linear regression baseline
            - **NLinearModel**: Normalized linear model
            - **DLinearModel**: Decomposition linear (trend + seasonal)
            - **xLSTMModel**: Extended LSTM with gating
            - **xLSTMTimeModel**: xLSTM with temporal attention
            - **TimeMixerModel**: Multi-scale mixing with decomposition (recommended)
            """
        )
        
        # Event handlers
        def update_download_button(df):
            # df will be a pandas DataFrame (or None). When it's not None, return a gr.update dict
            if df is None or getattr(df, "empty", False):
                return gr.update(visible=False, value=None)

            # Save CSV to a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
                df.to_csv(tmp.name, index=False)
                temp_path = tmp.name

            # Return the file path for DownloadButton
            return gr.update(visible=True, value=temp_path)

        
        predict_btn.click(
            fn=forecast_stock,
            inputs=[csv_input, model_dropdown, forecast_days_slider, smoothing_checkbox],
            outputs=[plot_output, predictions_output, status_output]
        )
        
        predictions_output.change(
            fn=update_download_button,
            inputs=[predictions_output],
            outputs=[download_btn]
        )
    
    return app


if __name__ == "__main__":
    logger.info("Starting Stock Price Forecasting Web Application")
    
    # Create and launch interface
    app = create_interface()
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )