import os, sys

import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import f1_score

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train and test data")
            X_train, y_train, X_test, y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "K-Neighbor Classifier": KNeighborsClassifier(),
                "Support Vector Classifier": SVC(),
                "AdaBoost": AdaBoostClassifier(),
                "Gradient Boosting": GradientBoostingClassifier()
            }

            params = {
                "Logistic Regression": {},

                "Decision Tree": {
                    'criterion' : ['gini', 'entropy', 'log_loss'],
                    'max_depth': [3,4,5,6]
                },

                "Random Forest":{
                    'criterion' : ['gini', 'entropy', 'log_loss'],
                    'max_depth': [3,4,5,6]
                },

                "K-Neighbor Classifier": {
                    'n_neighbors': [3,5,7,9]
                },

                "Support Vector Classifier": {
                    'kernel': ['linear','poly','sigmoid','rbf']
                },

                "AdaBoost":{
                    'n_estimators': [30,50,80],
                    'learning_rate': [0.1, 0.5, 0.05]
                },

                "Gradient Boosting":{
                    'learning_rate': [0.1, 0.5, 0.05]
                }
            }

            model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test, models=models, param=params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score <0.6:
                raise CustomException("Best model not found")
            logging.info("Best model found on both training and testing set")

            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            f1_scoring = f1_score(y_test, predicted)
            return f1_scoring
            


        except Exception as e:
            raise CustomException(e, sys)