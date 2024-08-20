import pandas as pd
import os
import logging
from config import CONFIG
from PredictionProject import PredictionProject

logging.basicConfig(level=logging.INFO)

def main():
    logging.info('----------------------- START ----------------------')
    
    # Load initial data
    df = pd.read_excel(os.path.join(CONFIG['BASE_DIR'], 'spreadsheet/Final_Weekly_2009_2021.xlsx'))

    # Create and run a prediction project for each target column
    for target in CONFIG['TARGET_COLUMN']:
        logging.info(f"\nProcessing target: {target}")
        
        # Create a PredictionProject instance for the current target column
        project = PredictionProject(df, target)
        
        # Run the project
        project.run()

    logging.info('----------------------- END ----------------------')

if __name__ == "__main__":
    main()

