import pytest
from utils.tools import dotdict


@pytest.fixture
def dummy_config():
    """Minimal configuration object used by model constructors."""
    cfg = dotdict({
        'enc_in': 1,
        'c_out': 1,
        'pred_len': 2,
        'seq_len': 2,
        'scale': 1,
        'dropout': 0.0,
        'squeeze': 1,
    })
    return cfg


@pytest.fixture
def dummy_args():
    """Minimal arguments object used by utility tests."""
    args = dotdict({
        'learning_rate': 0.1,
        'lradj': None,
        'train_epochs': 10,
        'warmup_epochs': 1,
    })
    return args
