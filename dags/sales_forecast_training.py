from datetime import datetime, timedelta
from airflow.decorators import dag, task
import pandas as pd
import sys
import os
import logging
import mlflow




logger = logging.getLogger(__name__)


sys.path.append("/usr/local/airflow/include")
from utils.data_generator import RealisticSalesDataGenerator
from ml_models.train_models import ModelTrainer
from utils.mlflow_utils import MLflowManager
from data_validation.validators import DataValidator


default_args = {
    'owner' : 'chandrahas',
    'depends_on_past' : False,
    'start_date' : datetime(2026, 8, 27),
    'retries' : 1,
    'retry_delay' : timedelta(minutes=1),
    'catchup' : False,
    'schedule' : '@weekly'  
}

@dag(
    default_args = default_args,
    description = 'Sales forecast Training DAG',
    tags = ['ml', 'training', 'sales_forecast', 'sales']
)
def sales_forecast_training():
    @task()
    def extract_data_task():
        

        data_output_dir = "/tmp/sales_data"

        generator = RealisticSalesDataGenerator(
            start_date = '2022-01-01',
            end_date = '2022-03-31'
        )

        print('Generating realist sales data')

        file_paths = generator.generate_sales_data(output_dir = data_output_dir)
        total_files = sum(len(path) for path in file_paths.values())
        print(f'Generated {total_files} files')


        for data_type, paths in file_paths.items():
            print(f'{data_type} : {len(paths)} files')

        return {
            'data_output_dir' : data_output_dir,
            'file_path' : file_paths,
            'total_files' : total_files
        }

    @task()
    def validate_data_task(extract_result):
        file_paths = extract_result['file_paths']
        total_rows = 0
        issues_found = []
        logger.info(f'validating {len(file_paths['sales'])} sales files')
        for i, sales_file in enumerate(file_paths['sales']):
            df = pd.read_parquet(sales_file)

            if i == 0:
                logger.info(f'sales data columns: {df.columns.to_list()}')

            if df.empty:
                issues_found.append(f'sales file {sales_file} is empty')
                continue

            required_cols = ['date', 'store_id', 'product_id', 'quantity_sold', 'revenue']

            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                issues_found.append(f'sales files {sales_file} missing columns {missing_cols}')
                continue

            total_rows = len(df)
            if df['quantity_sold'].min() < 0:
                issues_found.append(f'sales file {sales_file} has negative quantity sold')

            if df['revenue'].min() < 0 :
                issues_found.append(f'sales file {sales_file} has negative revenue')

            for data_type in ['promotions', 'customer_traffic', 'store_events']:
                if data_type in file_paths and file_paths[data_type]: # checking for data types
                    sample_file = file_paths[data_type][0]
                    df = pd.read_parquet(sample_file)
                    logger.info(f'{data_type} data shape: {df.shape}')
                    logger.info(f'{data_type} data columns {df.columns.to_list()}')


            validation_summary = {
                'total_files_validated' : len(file_paths['sales'][:10]),
                'total_rows' : total_rows,
                'issues_found' : issues_found,
                'issues_count' : issues_found[:5],
                'file_paths' : file_paths
            }

            if issues_found:
                logger.error(f'valadition summary : {validation_summary}')
                for issue in issues_found:
                    logger.error(issue)
                raise Exception('valadition issue found')
            else:
                logger.info(f'validation_summary: {validation_summary}')

            return validation_summary    

    @task()
    def train_models_task(validation_summary):
        file_paths = validation_summary['file_paths']
        logger.info(f"training models....")
        sales_df = []
        max_files = 50
        for i, sales_file in enumerate(file_paths['sales'][:max_files]):
            df = pd.read_parquet(sales_file)
            sales_df.append(df)
            if (i+1) % 10 == 0:
                logger.info(f'loaded {i+1} sales files')

        sales_df = pd.concat(sales_df, ignore_index=True)
        print(f'combained sales data shape {sales_df.shape}')

        daily_df = (sales_df.groupby(['sales','store_id','product_id','category'])
                                     .agg({
                                         'quantity_sold' : 'sum',
                                         'revenue' : 'sum',
                                         'cost' : 'sum',
                                         'profit' : 'sum',
                                         'discount_percennt' : 'mean',
                                         'unit_price' : 'mean'
                                     }).reset_index()
                                     )

        daily_sales = daily_sales.rename(columns = {'reveue' : 'sales'})
        if file_paths.get('promotions'):
            promo_df = pd.read_parquet(file_paths['promotions'][0])
            promo_summary = (
                promo_df.groupby(['date','product_id'])['discount_percent'].max().reset_index()
            )

            promo_summary['has_promotion'] = 1


            daily_sales  = daily_sales.merge(
                promo_summary[['date','product_id','has_promotion']],
                on = ['date', 'product_id'],
                how = 'left'
                )
            daily_sales['has_promotion'] = daily_sales['has_promotion'].fillna(0).astype(int)

        if file_paths.get('customer_traffic'):
            traffic_df = []
            for traffic_file in file_paths['customer_traffic'][:10]:
                traffic_dfs = traffic_df.append(pd.read_parquet(traffic_file))

            traffic_dfs = pd.concat(traffic_dfs, ignore_index=True)
            traffic_summary = (
                traffic_df.groupby(['date', 'store_id'])
                .agg({'customer_traffic' : 'sum', 'is_holiday' : max})
                .reset_index()
            )
            daily_sales = daily_sales.merge(
                traffic_summary,
                on = ['date', 'store_id'],
                how = 'left'
            )

        logger.info(f'Final training data shape : {daily_sales.shape}')
        logger.info(f'Columns : {daily_sales.columns.tolist()}')


        trainer = ModelTrainer()
        store_daily_sales = (
            daily_sales.gropuby(['date','store_id'])
            .agg({'sales' : 'sum',
                   'has_promotion' : 'mean',
                   'quantity_sold' : 'sum',
                   'profit' : 'sum',
                   'customer_traffic': 'first',
                   'is_holiday' : 'first'
                   })
        )
        store_daily_sales[date] = pd.to_datetime(store_daily_sales['date'])

        train_df, val_df, test_df = trainer.prepare_data(
            store_daily_sales, 
            target_col = 'sales',
            date_col = 'date',
            group_cols = ['store_id'],
            categorical_cols = ['store_id']
        )

        logger.info(
            f"Train shape {train_df_df.shape}, val_shape : {val_df.shape}, test shape : {test_df.shape}"
        )

        results = trainer.train_all_models(train_df, val_df, test_df, target_col = 'sales', use_optuna = True)

        for model_name, model_results in results.items():
            # logger.info(f"Model {model_name}, results : {model_results}")
            if "metrics" in model_results:
                print(f"\n{model_name} metrics: ")
                for metric, value in model_results['metrics'].items():
                    print(f" {metric} : {value:.4f}")

        print("\nVisualization charts have been generated and saved to MLflow/MinIO")
        print("Charts include:")
        print("  - Model metrics comparison")
        print("  - Predictions vs actual values")
        print("  - Residuals analysis")
        print("  - Error distribution")
        print("  - Feature importance comparison")

        serializable_results = {}

        for model_name, model_results in results.items():
            serializable_results[model_name] = {
                "metrics" : model_results.get("metrics", {})
            }

        current_run_id = (
            mlflow.active_run().info.run_id if mlflow.active_run() else None
        )

        return {
            'training_results' : serializable_results,
            'mlflow_run_id' : current_run_id
        }

    @task()
    def evaluate_models_task(training_results):
        results = training_results['training_results']
        mlflow_manager = MLflowManager()
        best_model_name = None 
        best_rmse = float('inf')
        for model_name, model_results in results.items():
            if 'metrics' in model_results and 'rmse' in model_results['metrics']:
                best_rmse = model_results['metrics']['rmse']
                best_model_name = model_name

        logger.info(f"Best model : {best_model_name}, with RMSE : {best_rmse}")
        best_run = mlflow_manager.get_best_model(metrics = 'rmse', ascending = True)

        return {
            'best_model' : best_model_name,
            'best_run_id' : best_run['run_id']
        }

    @task()
    def register_best_model_task(evaluation_results):
        evaluation_result = evaluation_results


            

    extract_result = extract_data_task()     
    validation_summary = validate_data_task(extract_result)


sales_forecast_training()   
