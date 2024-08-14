import pytest
import pandas as pd
import numpy as np
from PredictionProject import PredictionProject
from DataPreprocessor import DataPreprocessor
from ResultAnalyzer import ResultAnalyzer
from Visualizer import Visualizer
from config import CONFIG

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'Date': pd.date_range(start='2020-01-01', periods=100),
        'target': np.random.rand(100),
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100)
    })

def test_prediction_project_initialization(sample_df):
    project = PredictionProject(sample_df, 'target')
    
    assert isinstance(project.preprocessor, DataPreprocessor)
    assert isinstance(project.result_analyzer, ResultAnalyzer)
    assert isinstance(project.visualizer, Visualizer)

@pytest.mark.parametrize("action", [True, False])
def test_prediction_project_run(sample_df, monkeypatch, action):
    # Mock the actions in CONFIG
    monkeypatch.setitem(CONFIG['ACTION'], "Train models", action)
    monkeypatch.setitem(CONFIG['ACTION'], "Save models", action)
    
    project = PredictionProject(sample_df, 'target')
    
    # Mock the methods that are called in the run method
    project.preprocessor.load_and_preprocess_data = lambda lag: sample_df
    project.preprocessor.split_data = lambda: (sample_df, sample_df['target'], sample_df, sample_df['target'])
    project.result_analyzer.find_global_best_configs = lambda *args: None
    project.result_analyzer.update_global_results = lambda *args: None
    project.result_analyzer.calc_scores = lambda *args: None
    project.result_analyzer.save_results_to_excel = lambda *args: None
    project.visualizer.plot_all_models_predictions = lambda *args: None
    project.visualizer.plot_all_models_mape = lambda *args: None
    
    # Run the project
    project.run()
    
    # If we've reached this point without any exceptions, the test passes
    assert True