import numpy as np
import pytest

from data_provider.data_loader import Dataset_flame


def test_cal_diff():
    data = np.array([[1, 2], [3, 5], [6, 9]])
    # reshape to S,H,W
    data3 = data.reshape(3, 1, 2)
    out = Dataset_flame.__cal_diff__(data3)
    # first element should equal first row
    assert out.shape == data3.shape
    assert np.array_equal(out[0], data3[0])
    assert np.array_equal(out[1], data3[1] - data3[0])


def test_cal_mean():
    data = np.ones((2, 3, 4))
    out = Dataset_flame.__cal_mean__(data)
    assert out.shape == data.shape
    assert np.all(out == 1)


def test_cal_total_mean():
    data = np.arange(8).reshape(2, 2, 2)
    out = Dataset_flame.__cal_total_mean__(data)
    assert out.shape == data.shape
    assert np.all(out == np.mean(data))


def test_feature_engineering_shapes():
    # create small theta/ustar/xi sequences
    theta = np.zeros((2, 3, 4))
    ustar = np.zeros((2, 3, 4))
    xi = np.zeros((2, 3, 4))
    feat = Dataset_flame.__feature_engineering__(theta, ustar, xi)
    # output shape should be S, C, H, W with C=35 from implementation
    assert feat.shape[0] == 2
    assert feat.shape[1] == 35
    assert feat.shape[2] == 3
    assert feat.shape[3] == 4
