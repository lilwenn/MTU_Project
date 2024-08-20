import pytest
from unittest.mock import patch, MagicMock, ANY
import pandas as pd
from PredictionProject import PredictionProject

@pytest.fixture
def sample_df():
    """Fixture to provide a sample DataFrame for testing."""
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [5, 4, 3, 2, 1],
        'target': [2, 3, 4, 5, 6],
        'Date': pd.date_range(start='2024-01-01', periods=5)
    })

@pytest.fixture
def config_mock():
    """Fixture to mock CONFIG."""
    with patch('PredictionProject.CONFIG', {
        'TARGET_DIR': {'target': 'target_dir'},
        'LAG_LIST': [1, 2, 3],
        'ACTION': {"Train models": True, "Save models": True},
        'MODELS': {'model1': MagicMock()},
        'FORECAST_WEEKS': 4,
        'BASE_DIR': '/base/dir',
        'NON_ML': {},
    }) as mock_config:
        yield mock_config

def test_initialization_success(sample_df, config_mock):
    with patch('PredictionProject.DataPreprocessor') as MockDataPreprocessor, \
         patch('PredictionProject.ResultAnalyzer') as MockResultAnalyzer, \
         patch('PredictionProject.Visualizer') as MockVisualizer:
        
        project = PredictionProject(sample_df, 'target')
        
        assert project.df.equals(sample_df)
        assert project.target_column == 'target'
        MockDataPreprocessor.assert_called_once_with(sample_df, 'target')
        MockResultAnalyzer.assert_called_once_with('target_dir')
        MockVisualizer.assert_called_once_with('target_dir')

def test_initialization_target_column_not_found(sample_df, config_mock):
    with pytest.raises(KeyError):
        PredictionProject(sample_df, 'non_existing_target')


def test_run(sample_df, config_mock):
    with patch('PredictionProject.DataPreprocessor') as MockDataPreprocessor, \
         patch('PredictionProject.ResultAnalyzer') as MockResultAnalyzer, \
         patch('PredictionProject.Visualizer') as MockVisualizer, \
         patch('PredictionProject.ModelTrainer') as MockModelTrainer:
        
        mock_preprocessor = MockDataPreprocessor.return_value
        mock_preprocessor.load_and_preprocess_data.return_value = sample_df
        mock_preprocessor.split_data.return_value = (sample_df[['feature1', 'feature2']], sample_df['target'], sample_df[['feature1', 'feature2']], sample_df['target'])
        
        mock_result_analyzer = MockResultAnalyzer.return_value
        mock_result_analyzer.find_global_best_configs.return_value = {'Predictions': sample_df['target']}
        mock_result_analyzer.calculate_weekly_mape = MagicMock()
        mock_result_analyzer.update_global_results = MagicMock()
        mock_result_analyzer.calc_scores = MagicMock()
        mock_result_analyzer.save_results_to_excel = MagicMock()
        
        mock_visualizer = MockVisualizer.return_value
        mock_visualizer.plot_all_models_predictions = MagicMock()
        mock_visualizer.plot_all_models_mape = MagicMock()
        
        project = PredictionProject(sample_df, 'target')
        project.run()
        
        # Vérifier que la méthode load_and_preprocess_data a été appelée avec chaque valeur de lag
        calls = [((lag,),) for lag in config_mock['LAG_LIST']]
        MockDataPreprocessor.return_value.load_and_preprocess_data.assert_has_calls(calls, any_order=True)
        
        # Vérifier que ModelTrainer a été appelé le nombre correct de fois
        assert MockModelTrainer.call_count == len(config_mock['LAG_LIST'])

        # Impression des appels réels pour débogage
        print("Actual calls to train_model:")
        for call in MockModelTrainer.return_value.train_model.call_args_list:
            print(call)

        # Vérifier que train_model a été appelé avec chaque lag en utilisant des assertions souples
        for lag in config_mock['LAG_LIST']:
            matched = any(
                args == ('model1', ANY, lag, 'target')
                for args, _ in MockModelTrainer.return_value.train_model.call_args_list
            )
            assert matched, f"Expected call with lag {lag} not found"

        # Vérifier les appels pour ResultAnalyzer et Visualizer
        MockResultAnalyzer.return_value.find_global_best_configs.assert_called_with('model1', config_mock['LAG_LIST'], {})
        MockResultAnalyzer.return_value.calculate_weekly_mape.assert_called_once()
        MockVisualizer.return_value.plot_all_models_predictions.assert_called_once()
        MockVisualizer.return_value.plot_all_models_mape.assert_called_once()
