# 📈 Stock Price Forecasting System

A production-grade, enterprise-level time series forecasting system for stock price prediction using state-of-the-art deep learning models. Features a user-friendly Gradio web interface and Docker deployment.

## 🎯 Features

- **6 Advanced Model Architectures**:
  - LinearModel: Simple linear regression baseline
  - NLinearModel: Normalized linear model
  - DLinearModel: Decomposition linear (trend + seasonal)
  - xLSTMModel: Extended LSTM with exponential gating
  - xLSTMTimeModel: xLSTM with temporal attention
  - TimeMixerModel: Multi-scale mixing with decomposition

- **Production-Ready MLOps**:
  - Modular, maintainable codebase
  - Configuration management (YAML)
  - Comprehensive logging
  - Early stopping with checkpointing
  - Model persistence - saves best version of each trained model
  - Automatic model loading for inference
  - Savitzky-Golay smoothing post-processing

- **User-Friendly Interface**:
  - Interactive Gradio web UI
  - Upload CSV, select model, generate forecasts
  - Interactive Plotly visualizations
  - Downloadable prediction results

- **Docker Deployment**:
  - Production-optimized Dockerfile
  - Docker Compose for easy setup
  - Health checks and resource limits

## 📁 Project Structure

```
.
├── configs/
│   └── config.yaml           # Configuration file
├── data/
│   └── FPT_train.csv         # Training data (example)
├── results/
│   ├── submission.csv        # Generated predictions
│   ├── training_results.csv  # Model performance metrics
│   ├── training.log          # Training logs
│   └── models/               # Trained model checkpoints
│       ├── LinearModel_best.pth
│       ├── NLinearModel_best.pth
│       ├── DLinearModel_best.pth
│       ├── xLSTMModel_best.pth
│       ├── xLSTMTimeModel_best.pth
│       └── TimeMixerModel_best.pth
├── src/
│   ├── __init__.py
│   ├── models.py             # Neural network architectures
│   ├── dataset.py            # Data loading and preprocessing
│   ├── train.py              # Training and evaluation logic
│   └── utils.py              # Utility functions
├── app.py                    # Gradio web interface
├── train_models.py           # Script to train all models
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker configuration
├── docker-compose.yml        # Docker Compose configuration
└── README.md                 # This file
```

## 🚀 Quick Start

### Option 1: Using Docker (Recommended)

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access the web interface
# Open browser to: http://localhost:7860
```

### Option 2: Local Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd stock-forecasting

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python app.py
```

## 📊 Usage

### Web Interface

1. **Upload CSV File**: 
   - Must contain columns: `time` (date) and `close` (price)
   - Minimum 400 rows for default configuration

2. **Select Model**:
   - Choose from 6 available architectures
   - TimeMixerModel recommended for best performance

3. **Configure Forecast**:
   - Set forecast days (1-365)
   - Enable/disable smoothing

4. **Generate Predictions**:
   - View interactive plot
   - Download predictions as CSV

### Training Models from Scratch

```bash
# Train all models and save best version of each
python train_models.py --data_path data/FPT_train.csv --config configs/config.yaml

# Models are automatically saved to results/models/
# - LinearModel_best.pth
# - NLinearModel_best.pth
# - DLinearModel_best.pth
# - xLSTMModel_best.pth
# - xLSTMTimeModel_best.pth
# - TimeMixerModel_best.pth
```

**Note**: The `--save_model` flag is enabled by default, so models are always saved after training. You can disable it with `--no-save_model` if needed.

### Model Workflow

1. **Training Phase**:
   - Trains all 6 models on your data
   - Evaluates on validation set with early stopping
   - Saves the best version of each model to `results/models/`
   - Generates predictions with the overall best model

2. **Inference Phase**:
   - Web app automatically discovers saved models in `results/models/`
   - When you select a model in the UI, it loads the corresponding trained weights
   - Uses the trained model for predictions instead of random weights
   - Dramatically improves prediction accuracy!

3. **Model Selection**:
   - Compare models side-by-side using `results/training_results.csv`
   - Each model's validation and test losses are logged
   - Choose the best performer for your use case

### Python API Usage

```python
from src.utils import load_config, set_seed, get_device
from src.dataset import load_and_preprocess_data, prepare_inference_data
from src.models import get_model
from src.train import predict

# Load configuration
config = load_config('configs/config.yaml')
set_seed(config['seed'])
device = get_device()

# Load and preprocess data
df = load_and_preprocess_data('data/stock_data.csv')

# Create model
model = get_model('TimeMixerModel', config)
model = model.to(device)

# Prepare input
input_data = prepare_inference_data(df, config['data']['input_len'])

# Make predictions
predictions = predict(model, input_data, device)
```

## ⚙️ Configuration

Edit `configs/config.yaml` to customize:

```yaml
# Random seed for reproducibility
seed: 42

# Data configuration
data:
  input_len: 400        # Input sequence length
  output_len: 100       # Forecast horizon
  train_ratio: 0.7      # Training data ratio
  val_ratio: 0.15       # Validation data ratio

# Training configuration
training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 0.001
  early_stopping_patience: 10

# Model-specific parameters
models:
  timemixer:
    hidden_size: 128
    n_scales: 3
    kernel_size: 25
```

## 📈 Model Performance

Models are evaluated using MSE (Mean Squared Error) on validation and test sets. The system automatically selects the best-performing model based on validation loss.

Example performance (FPT stock data):
- TimeMixerModel: Val Loss ~0.0012
- xLSTMTimeModel: Val Loss ~0.0015
- DLinearModel: Val Loss ~0.0018

## 🛠️ Development

### Adding a New Model

1. Define model class in `src/models.py`:

```python
class MyNewModel(nn.Module):
    def __init__(self, input_len, output_len):
        super().__init__()
        # Define layers
    
    def forward(self, x):
        # Forward pass
        return output
```

2. Add to `get_model()` factory function
3. Update configuration in `config.yaml`
4. Update `MODEL_OPTIONS` in `app.py`

### Running Tests

```bash
# Run unit tests (if implemented)
pytest tests/

# Check code quality
flake8 src/
black src/ --check
```

## 📦 Docker Details

### Building the Image

```bash
docker build -t stock-forecasting:latest .
```

### Running the Container

```bash
docker run -p 7860:7860 -v $(pwd)/data:/app/data stock-forecasting:latest
```

### Docker Compose Services

- **stock-forecasting**: Main application service
- Exposed on port 7860
- Persistent volumes for data, models, and results
- Health checks enabled
- Resource limits configured

## 🔧 Troubleshooting

### Common Issues

1. **Insufficient Data Error**:
   - Ensure CSV has at least `input_len` rows (default: 400)

2. **Out of Memory**:
   - Reduce `batch_size` in config
   - Use CPU instead of CUDA
   - Reduce model `hidden_size`

3. **Port Already in Use**:
   - Change port in `docker-compose.yml` or `app.py`

4. **CSV Format Issues**:
   - Ensure columns are named `time` and `close`
   - Check date format is parseable

## 📝 Data Format

Required CSV structure:

```csv
time,close
2023-01-01,100.5
2023-01-02,101.2
2023-01-03,99.8
...
```

- `time`: Date column (YYYY-MM-DD format)
- `close`: Closing price (numeric)

## 📄 License

This project is licensed under the MIT License.

---

**Note**: This system is for educational and research purposes. Always validate predictions before making financial decisions.