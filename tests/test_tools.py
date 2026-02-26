import os
import numpy as np
import torch
import pytest

from utils import tools


def test_dotdict_attribute_access():
    d = tools.dotdict({'a': 1, 'b': 2})
    assert d.a == 1
    assert d['b'] == 2
    d.c = 3
    assert d['c'] == 3


def test_standard_scaler_roundtrip():
    scaler = tools.StandardScaler(mean=2.0, std=4.0)
    data = np.array([2.0, 6.0])
    transformed = scaler.transform(data)
    inv = scaler.inverse_transform(transformed)
    assert np.allclose(inv, data)


def test_adjust_learning_rate_type1(dummy_args):
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.randn(1))], lr=0.1)
    dummy_args.lradj = 'type1'
    tools.adjust_learning_rate(optimizer, epoch=1, args=dummy_args)
    # type1 lowers lr by factor 0.5 every epoch
    assert optimizer.param_groups[0]['lr'] == pytest.approx(0.1)
    tools.adjust_learning_rate(optimizer, epoch=2, args=dummy_args)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(0.05)


def test_adjust_learning_rate_type2(dummy_args):
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.randn(1))], lr=0.1)
    dummy_args.lradj = 'type2'
    # epoch not in schedule should not change lr
    tools.adjust_learning_rate(optimizer, epoch=1, args=dummy_args)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(0.1)
    tools.adjust_learning_rate(optimizer, epoch=2, args=dummy_args)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(5e-5)


def test_adjust_learning_rate_cosine(dummy_args):
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.randn(1))], lr=0.1)
    dummy_args.lradj = 'cosine'
    # epoch is within training range
    tools.adjust_learning_rate(optimizer, epoch=5, args=dummy_args)
    expected = dummy_args.learning_rate / 2 * (1 + np.cos(5 / dummy_args.train_epochs * np.pi))
    assert optimizer.param_groups[0]['lr'] == pytest.approx(expected)


def test_early_stopping(tmp_path):
    # use a small dummy model
    model = torch.nn.Linear(1, 1)
    es = tools.EarlyStopping(patience=2, verbose=False)
    path = str(tmp_path)
    # first call should save checkpoint
    es(1.0, model, path)
    assert os.path.exists(os.path.join(path, 'checkpoint.pth'))
    # next two calls with equal loss should increment counter and then early_stop
    es(1.0, model, path)
    assert not es.early_stop
    es(1.0, model, path)
    assert es.early_stop


def test_adjustment():
    gt = np.array([0, 1, 1, 0, 1, 1, 1, 0])
    pred = np.array([0, 1, 0, 0, 1, 0, 1, 0])
    new_gt, new_pred = tools.adjustment(gt.copy(), pred.copy())
    # pred should fill in the gaps around anomalies in gt
    assert new_pred[2] == 1
    assert new_pred[5] == 1


def test_cal_accuracy():
    y_pred = np.array([1, 0, 1])
    y_true = np.array([1, 1, 1])
    assert tools.cal_accuracy(y_pred, y_true) == pytest.approx(2 / 3)
