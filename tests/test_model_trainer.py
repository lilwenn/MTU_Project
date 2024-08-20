import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from PredictionProject import PredictionProject
from config import CONFIG

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'Date': pd.date_range(start='2020-01-01', periods=100),
        'litres': np.random.rand(100) * 1000,
        'num_suppliers': np.random.randint(50, 100, 100),
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
    })
"""
@pytest.fixture
def prediction_project(sample_df):
    return PredictionProject(sample_df, 'litres')

def test_initialization(prediction_project):
    assert isinstance(prediction_project.df, pd.DataFrame)
    assert prediction_project.target_column == 'litres'
    assert prediction_project.preprocessor is not None
    assert prediction_project.result_analyzer is not None
    assert prediction_project.visualizer is not None

@patch('PredictionProject.DataPreprocessor')
def test_run_preprocessing(mock_preprocessor, prediction_project):
    mock_preprocessor.return_value.load_and_preprocess_data.return_value = pd.DataFrame()
    mock_preprocessor.return_value.split_data.return_value = (pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.Series())
    
    prediction_project.run()
    
    assert mock_preprocessor.return_value.load_and_preprocess_data.called
    assert mock_preprocessor.return_value.split_data.called

@patch('PredictionProject.ModelTrainer')
def test_run_model_training(mock_trainer, prediction_project):
    with patch.object(prediction_project.preprocessor, 'load_and_preprocess_data', return_value=pd.DataFrame()):
        with patch.object(prediction_project.preprocessor, 'split_data', return_value=(pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.Series())):
            prediction_project.run()
    
    assert mock_trainer.return_value.train_model.called

@patch('PredictionProject.ResultAnalyzer')
def test_run_result_analysis(mock_analyzer, prediction_project):
    with patch.object(prediction_project.preprocessor, 'load_and_preprocess_data', return_value=pd.DataFrame()):
        with patch.object(prediction_project.preprocessor, 'split_data', return_value=(pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.Series())):
            prediction_project.run()
    
    assert mock_analyzer.return_value.find_global_best_configs.called
    assert mock_analyzer.return_value.calculate_weekly_mape.called
    assert mock_analyzer.return_value.update_global_results.called

@patch('PredictionProject.Visualizer')
def test_run_visualization(mock_visualizer, prediction_project):
    with patch.object(prediction_project.preprocessor, 'load_and_preprocess_data', return_value=pd.DataFrame()):
        with patch.object(prediction_project.preprocessor, 'split_data', return_value=(pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.Series())):
            prediction_project.run()
    
    assert mock_visualizer.return_value.plot_all_models_predictions.called
    assert mock_visualizer.return_value.plot_all_models_mape.called

def test_run_with_different_target(sample_df):
    project = PredictionProject(sample_df, 'num_suppliers')
    assert project.target_column == 'num_suppliers'

@pytest.mark.parametrize("lag", CONFIG['LAG_LIST'])
def test_run_with_different_lags(lag, prediction_project):
    with patch.object(prediction_project.preprocessor, 'load_and_preprocess_data', return_value=pd.DataFrame()) as mock_preprocess:
        with patch.object(prediction_project.preprocessor, 'split_data', return_value=(pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.Series())):
            prediction_project.run()
    
    mock_preprocess.assert_called_with(lag)

@patch('PredictionProject.ModelTrainer')
def test_run_without_training(mock_trainer, prediction_project):
    CONFIG['ACTION']["Train models"] = False
    with patch.object(prediction_project.preprocessor, 'load_and_preprocess_data', return_value=pd.DataFrame()):
        with patch.object(prediction_project.preprocessor, 'split_data', return_value=(pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.Series())):
            prediction_project.run()
    
    assert not mock_trainer.return_value.train_model.called
    CONFIG['ACTION']["Train models"] = True  # Reset to original value

@patch('PredictionProject.ResultAnalyzer')
def test_run_without_saving(mock_analyzer, prediction_project):
    CONFIG['ACTION']["Save models"] = False
    with patch.object(prediction_project.preprocessor, 'load_and_preprocess_data', return_value=pd.DataFrame()):
        with patch.object(prediction_project.preprocessor, 'split_data', return_value=(pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.Series())):
            prediction_project.run()
    
    assert not mock_analyzer.return_value.find_global_best_configs.called
    CONFIG['ACTION']["Save models"] = True  # Reset to original value

def test_run_with_empty_dataframe():
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="The DataFrame cannot be empty."):
        PredictionProject(empty_df, 'litres')

def test_run_with_missing_target_column(sample_df):
    with pytest.raises(KeyError, match="Target column 'non_existent_column' not found in DataFrame."):
        PredictionProject(sample_df, 'non_existent_column')

@patch('PredictionProject.ModelTrainer')
def test_run_with_exception_in_training(mock_trainer, prediction_project):
    mock_trainer.return_value.train_model.side_effect = Exception("Training failed")
    with patch.object(prediction_project.preprocessor, 'load_and_preprocess_data', return_value=pd.DataFrame()):
        with patch.object(prediction_project.preprocessor, 'split_data', return_value=(pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.Series())):
            with pytest.raises(Exception, match="Training failed"):
                prediction_project.run()

def test_run_integration(prediction_project):
    try:
        prediction_project.run()
    except Exception as e:
        pytest.fail(f"run() raised {e} unexpectedly!")
"""