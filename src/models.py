from typing import Tuple
import torch
import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self, input_len: int, output_len: int) -> None:
        super(LinearModel, self).__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.linear = nn.Linear(input_len, output_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class NLinearModel(nn.Module):
    def __init__(self, input_len: int, output_len: int) -> None:
        super(NLinearModel, self).__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.linear = nn.Linear(input_len, output_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last_value = x[:, -1:].detach()
        x_norm = x - last_value
        output_norm = self.linear(x_norm)
        output = output_norm + last_value
        return output


class DLinearModel(nn.Module):
    def __init__(self, input_len: int, output_len: int, kernel_size: int) -> None:
        super(DLinearModel, self).__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.kernel_size = kernel_size
        self.linear_trend = nn.Linear(input_len, output_len)
        self.linear_seasonal = nn.Linear(input_len, output_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = self._moving_average(x, self.kernel_size)
        seasonal = x - trend
        trend_output = self.linear_trend(trend)
        seasonal_output = self.linear_seasonal(seasonal)
        output = trend_output + seasonal_output
        return output

    def _moving_average(self, x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        batch_size, seq_len = x.shape
        pad_size = (kernel_size - 1) // 2
        x_padded = torch.cat([
            x[:, 0:1].repeat(1, pad_size),
            x,
            x[:, -1:].repeat(1, pad_size)
        ], dim=1)
        x_unfolded = x_padded.unfold(dimension=1, size=kernel_size, step=1)
        trend = x_unfolded.mean(dim=-1)
        return trend


class xLSTMModel(nn.Module):
    def __init__(
        self,
        input_len: int,
        output_len: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ) -> None:
        super(xLSTMModel, self).__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.exp_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

        self.fc = nn.Linear(hidden_size, output_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = x.unsqueeze(-1)  # [batch, seq_len, 1]

        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]  # [batch, hidden_size]

        gate = self.exp_gate(last_hidden)
        gated_hidden = last_hidden * gate

        output = self.fc(gated_hidden)  # [batch, output_len]
        return output


class xLSTMTimeModel(nn.Module):
    def __init__(
        self,
        input_len: int,
        output_len: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.2
    ) -> None:
        super(xLSTMTimeModel, self).__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.hidden_size = hidden_size

        self.pos_encoding = nn.Parameter(torch.randn(1, input_len, hidden_size))
        self.input_proj = nn.Linear(1, hidden_size)

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.exp_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = x.unsqueeze(-1)  # [batch, seq_len, 1]

        x = self.input_proj(x)  # [batch, seq_len, hidden_size]
        x = x + self.pos_encoding

        lstm_out, (h_n, c_n) = self.lstm(x)
        attn_out, _ = self.temporal_attention(lstm_out, lstm_out, lstm_out)

        combined = lstm_out + attn_out
        combined = self.layer_norm(combined)

        last_hidden = combined[:, -1, :]  # [batch, hidden_size]

        gate = self.exp_gate(last_hidden)
        gated_hidden = last_hidden * gate

        output = self.fc(gated_hidden)  # [batch, output_len]
        return output


class TimeMixerModel(nn.Module):
    def __init__(
        self,
        input_len: int,
        output_len: int,
        hidden_size: int = 128,
        n_scales: int = 3,
        kernel_size: int = 25
    ) -> None:
        super(TimeMixerModel, self).__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.hidden_size = hidden_size
        self.n_scales = n_scales
        self.kernel_size = kernel_size

        self.scale_mixers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_len, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, hidden_size)
            ) for _ in range(n_scales)
        ])

        self.trend_extractor = nn.Sequential(
            nn.Linear(input_len, hidden_size),
            nn.Tanh()
        )

        self.seasonal_extractor = nn.Sequential(
            nn.Linear(input_len, hidden_size),
            nn.ReLU()
        )

        self.scale_attention = nn.Sequential(
            nn.Linear(hidden_size * n_scales, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, n_scales),
            nn.Softmax(dim=-1)
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, output_len)
        )

    def _moving_average(self, x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        batch_size, seq_len = x.shape
        pad_size = (kernel_size - 1) // 2
        x_padded = torch.cat([
            x[:, 0:1].repeat(1, pad_size),
            x,
            x[:, -1:].repeat(1, pad_size)
        ], dim=1)
        x_unfolded = x_padded.unfold(dimension=1, size=kernel_size, step=1)
        trend = x_unfolded.mean(dim=-1)
        return trend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        trend = self._moving_average(x, self.kernel_size)
        seasonal = x - trend

        trend_feat = self.trend_extractor(trend)
        seasonal_feat = self.seasonal_extractor(seasonal)

        scale_outputs = []
        for i, mixer in enumerate(self.scale_mixers):
            if i == 0:
                scale_input = x
            else:
                scale_factor = 2 ** i
                if x.size(1) >= scale_factor:
                    scale_input = x[:, ::scale_factor]
                    if scale_input.size(1) < self.input_len:
                        pad_size = self.input_len - scale_input.size(1)
                        scale_input = torch.cat([
                            scale_input,
                            scale_input[:, -1:].repeat(1, pad_size)
                        ], dim=1)
                else:
                    scale_input = x

            scale_out = mixer(scale_input)
            scale_outputs.append(scale_out)

        all_scales = torch.stack(scale_outputs, dim=1)
        scale_concat = all_scales.view(batch_size, -1)
        scale_weights = self.scale_attention(scale_concat)
        scale_weights = scale_weights.unsqueeze(-1)

        mixed_scales = (all_scales * scale_weights).sum(dim=1)
        combined = torch.cat([trend_feat + seasonal_feat, mixed_scales], dim=-1)

        output = self.fusion(combined)
        return output


def get_model(model_name: str, config: dict) -> nn.Module:
    input_len = config['data']['input_len']
    output_len = config['data']['output_len']
    
    if model_name == "LinearModel":
        return LinearModel(input_len, output_len)
    elif model_name == "NLinearModel":
        return NLinearModel(input_len, output_len)
    elif model_name == "DLinearModel":
        kernel_size = config['models']['dlinear']['kernel_size']
        return DLinearModel(input_len, output_len, kernel_size)
    elif model_name == "xLSTMModel":
        params = config['models']['xlstm']
        return xLSTMModel(
            input_len, output_len,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params['dropout']
        )
    elif model_name == "xLSTMTimeModel":
        params = config['models']['xlstm_time']
        return xLSTMTimeModel(
            input_len, output_len,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            n_heads=params['n_heads'],
            dropout=params['dropout']
        )
    elif model_name == "TimeMixerModel":
        params = config['models']['timemixer']
        return TimeMixerModel(
            input_len, output_len,
            hidden_size=params['hidden_size'],
            n_scales=params['n_scales'],
            kernel_size=params['kernel_size']
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")