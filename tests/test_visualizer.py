import pytest
import os
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from unittest.mock import mock_open
from Visualizer import Visualizer
from config import CONFIG

# Set matplotlib to use the 'Agg' backend to avoid GUI operations
@pytest.fixture(scope="module", autouse=True)
def set_matplotlib_backend():
    matplotlib.use('Agg')
    yield
    plt.close('all')  # Close all figures after tests

@pytest.fixture
def visualizer():
    return Visualizer("test_target_dir")

@pytest.fixture
def sample_data():
    dates = pd.date_range(start="2023-01-01", periods=52, freq="W")
    y_true = np.random.rand(52)
    y_pred = np.random.rand(52)
    return dates, y_true, y_pred

def test_init(visualizer):
    assert visualizer.target_dir == "test_target_dir"

def test_plot_predictions_by_model(visualizer, sample_data, monkeypatch):
    dates, y_true, y_pred = sample_data
    
    # Mock plt.savefig to avoid actually saving the plot
    def mock_savefig(*args, **kwargs):
        pass
    
    monkeypatch.setattr(plt, "savefig", mock_savefig)
    
    visualizer.plot_predictions_by_model(y_pred, dates, y_true, "TestModel")
    
    # Assert that the plot was "saved"
    assert plt.gcf().number == 1

def test_plot_all_models_mape(visualizer, sample_data, monkeypatch):
    dates, _, _ = sample_data
    
    # Mock open to avoid actual file operations
    mock_open_func = mock_open(read_data=json.dumps({f"week_{i}": np.random.rand() for i in range(1, 53)}))
    monkeypatch.setattr("builtins.open", mock_open_func)
    
    # Mock json.load to return sample data
    def mock_json_load(f):
        return json.loads(f.read())
    
    monkeypatch.setattr(json, "load", mock_json_load)
    
    # Mock plt.savefig to avoid actually saving the plot
    def mock_savefig(*args, **kwargs):
        pass
    
    monkeypatch.setattr(plt, "savefig", mock_savefig)
    
    visualizer.plot_all_models_mape(dates)
    
    # Assert that the plot was "saved"
    assert plt.gcf().number == 1

def test_plot_weekly_mape(visualizer, sample_data, monkeypatch):
    dates, _, _ = sample_data
    weekly_mape = {f"week_{i}": np.random.rand() for i in range(1, 53)}
    
    # Mock plt.savefig to avoid actually saving the plot
    def mock_savefig(*args, **kwargs):
        pass
    
    monkeypatch.setattr(plt, "savefig", mock_savefig)
    
    visualizer.plot_weekly_mape(dates, weekly_mape, "TestModel")
    
    # Assert that the plot was "saved"
    assert plt.gcf().number == 1
    
def test_plot_all_models_predictions_file_not_found(visualizer, sample_data, monkeypatch):
    dates, y_true, _ = sample_data
    
    # Mock the open function to raise FileNotFoundError
    mock_open_func = mock_open()
    mock_open_func.side_effect = FileNotFoundError
    
    monkeypatch.setattr("builtins.open", mock_open_func)
    
    with pytest.raises(FileNotFoundError):
        visualizer.plot_all_models_predictions(dates, y_true)

def test_plot_all_models_mape_file_not_found(visualizer, sample_data, monkeypatch):
    dates, _, _ = sample_data
    
    # Mock the open function to raise FileNotFoundError
    mock_open_func = mock_open()
    mock_open_func.side_effect = FileNotFoundError
    
    monkeypatch.setattr("builtins.open", mock_open_func)
    
    with pytest.raises(FileNotFoundError):
        visualizer.plot_all_models_mape(dates)
