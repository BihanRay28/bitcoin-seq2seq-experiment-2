from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import load_configuration


@pytest.fixture
def cfg():
    config, _ = load_configuration()
    config.encoder_hidden_size = 8
    config.decoder_hidden_size = 8
    config.attention_dim = 8
    config.head_hidden_size = 8
    config.dropout = 0.0
    config.batch_size = 16
    config.device = 'cpu'
    torch.set_num_threads(2)
    return config


@pytest.fixture
def frame():
    rng = np.random.default_rng(51); n = 1800
    previous = 30000 * np.exp(np.cumsum(rng.normal(0,.001,n)))
    o = previous * np.exp(rng.normal(0,.0001,n))
    c = o * np.exp(rng.normal(0,.002,n))
    h = np.maximum(o,c)*np.exp(rng.uniform(0,.004,n))
    l = np.minimum(o,c)*np.exp(-rng.uniform(0,.004,n))
    v = rng.uniform(0,500,n); v[0] = 0
    return pd.DataFrame({'datetime':pd.date_range('2024-01-01',periods=n,freq='30min'),
                         'open':o,'high':h,'low':l,'close':c,'volume':v})
