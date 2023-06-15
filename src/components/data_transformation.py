import os, sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            cat_nominal_columns = ['Education', 'Gender', 'Married', 'Property_Area', 'Self_Employed']
            cat_ordinal_column = ['Dependents']
            num_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount','Loan_Amount_Term', 'Credit_History']
            target_feature = ['Loan_Status']

            num_pipeline = Pipeline([
                ('si', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler(with_mean=False))
            ])

            cat_ordinal_pipeline = Pipeline([
                ('si', SimpleImputer(strategy='most_frequent')),
                ('oe', OrdinalEncoder(categories=[['0','1','2','3+']])),
                ('ss', StandardScaler(with_mean=False))
                ])
            
            cat_nominal_pipeline = Pipeline([
                ('si', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder()),
                ('ss', StandardScaler(with_mean=False))
                ])
    
            
            logging.info(f"Categorical ordinal columns: {cat_ordinal_column}")
            logging.info(f"Categorical nominal column: {cat_nominal_columns}")
            logging.info(f"Numerical columns: {num_columns}")

            preprocessor = ColumnTransformer([
                ('cat_ordinal_pipeline', cat_ordinal_pipeline, cat_ordinal_column),
                ('cat_nominal_pipeline', cat_nominal_pipeline, cat_nominal_columns),
                ('num_pipeline', num_pipeline, num_columns)
            ])

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()


            logging.info("Applying preprocessor object to train and test dataframe")

            target_column = ['Loan_Status']

            input_feature_train_df = train_df.drop(columns=target_column).drop(columns=['Loan_ID'])
            target_feature_train_df = train_df['Loan_Status']

            input_feature_test_df = test_df.drop(columns=target_column).drop(columns=['Loan_ID'])
            target_feature_test_df = test_df['Loan_Status']

            train_arr = preprocessing_obj.fit_transform(train_df)
            test_arr = preprocessing_obj.transform(test_df)

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            le = LabelEncoder()   # For encoding target feature
            target_feature_train_arr =le.fit_transform(target_feature_train_df) 
            target_feature_test_arr = le.transform(target_feature_test_df)

            input_feature_train_transformed_df = pd.DataFrame(input_feature_train_arr)
            input_feature_train_transformed_df['target'] = target_feature_train_arr
            train_arr = np.array(input_feature_train_transformed_df)

            input_feature_test_transformed_df = pd.DataFrame(input_feature_test_arr)
            input_feature_test_transformed_df['target'] = target_feature_test_arr
            test_arr = np.array(input_feature_test_transformed_df)

            logging.info('Saved preprocessing object')

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)