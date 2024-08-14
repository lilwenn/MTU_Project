import pytest
import json
import os
import pandas as pd
import numpy as np
from ResultAnalyzer import ResultAnalyzer
from config import CONFIG

@pytest.fixture
def sample_results(tmp_path):
    results = {
        'StandardScaler': {
            'f_regression': {
                'MAPE_Score': 5.0,
                'Execution Time': 1.0,
                'Mean Train Score': -0.1,
                'Mean Test Score': -0.2,
                'Predictions': [1.0, 2.0, 3.0],
                'Selected Features': ['feature1', 'feature2'],
                'Best Parameters': {'param1': 1, 'param2': 2}
            }
        }
    }
    
    model_dir = tmp_path / "result" / "target_results" / "train_by_model"
    model_dir.mkdir(parents=True)
    
    for lag in CONFIG['LAG_LIST']:
        with open(model_dir / f"TestModel_{lag}_model_{CONFIG['FORECAST_WEEKS']}.json", 'w') as f:
            json.dump(results, f)
    
    return tmp_path

def test_find_global_best_configs(sample_results):
    analyzer = ResultAnalyzer('target_results')
    analyzer.target_dir = sample_results / "result" / "target_results"
    
    best_config = analyzer.find_global_best_configs('TestModel', CONFIG['LAG_LIST'], [])
    
    assert best_config is not None
    assert best_config['MAPE_Score'] == 5.0
    assert 'Execution Time' in best_config
    assert 'Selected Features' in best_config

def test_update_global_results(sample_results):
    analyzer = ResultAnalyzer('target_results')
    analyzer.target_dir = sample_results / "result" / "target_results"
    
    analyzer.find_global_best_configs('TestModel', CONFIG['LAG_LIST'], [])
    analyzer.update_global_results('TestModel')
    
    global_results_file = sample_results / "result" / "target_results" / "global_results.json"
    assert global_results_file.exists()
    
    with open(global_results_file, 'r') as f:
        global_results = json.load(f)
    
    assert 'TestModel' in global_results

def test_calc_scores(sample_results):
    analyzer = ResultAnalyzer('target_results')
    analyzer.target_dir = sample_results / "result" / "target_results"
    
    y_true = np.array([1.0, 2.0, 3.0])
    
    analyzer.find_global_best_configs('TestModel', CONFIG['LAG_LIST'], [])
    analyzer.update_global_results('TestModel')
    result = analyzer.calc_scores(y_true)
    
    assert 'TestModel' in result
    assert 'MAPE' in result['TestModel']
    assert 'MAE' in result['TestModel']
    assert 'RMSE' in result['TestModel']

def test_save_results_to_excel(sample_results, tmp_path):
    analyzer = ResultAnalyzer('target_results')
    analyzer.target_dir = sample_results / "result" / "target_results"
    
    output_file = tmp_path / "test_results.xlsx"
    
    analyzer.find_global_best_configs('TestModel', CONFIG['LAG_LIST'], [])
    analyzer.update_global_results('TestModel')
    analyzer.calc_scores(np.array([1.0, 2.0, 3.0]))
    analyzer.save_results_to_excel(output_file, CONFIG['FORECAST_WEEKS'])
    
    assert output_file.exists()
    df = pd.read_excel(output_file)
    assert 'Model Name' in df.columns
    assert 'MAPE' in df.columns
    assert 'MAE' in df.columns