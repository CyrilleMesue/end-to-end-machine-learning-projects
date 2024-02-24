import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostRegressor, AdaBoostClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    RandomForestRegressor, RandomForestClassifier
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from xgboost import XGBRegressor, XGBClassifier 
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models, load_json

@dataclass
class ModelTrainerConfig:
    basename_tag = "*"
    trained_model_file_path=os.path.join("artifacts",f"{basename_tag}model.pkl")

class ModelTrainer:
    def __init__(self,
                 modeling_type = "reg"
                 ):
        self.model_trainer_config=ModelTrainerConfig()
        self.modeling_type = modeling_type


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            if self.modeling_type == "reg":
                models = {
                    "Random Forest": RandomForestRegressor(random_state = 32),
                    "Decision Tree": DecisionTreeRegressor(random_state = 32),
                    "Gradient Boosting": GradientBoostingRegressor(random_state = 32),
                    "Linear Regression": LinearRegression(),
                    "XGBRegressor": XGBRegressor(random_state = 32),
                    "CatBoosting Regressor": CatBoostRegressor(random_state = 32,verbose=False),
                    "AdaBoost Regressor": AdaBoostRegressor(random_state = 32),
                }
            else:
                models = {
                    "Random Forest": RandomForestClassifier(random_state = 32),
                    "Decision Tree": DecisionTreeClassifier(random_state = 32),
                    "Gradient Boosting": GradientBoostingClassifier(random_state = 32),
                    "Logistic Regression": LogisticRegression(random_state = 32),
                    "XGBClassifier": XGBClassifier(random_state = 32),
                    "CatBoosting Classifier": CatBoostClassifier(random_state = 32,verbose=False),
                    "AdaBoost Classifier": AdaBoostClassifier(random_state = 32),
                    "MLPClassifier": MLPClassifier(random_state = 32, verbose=False),
                    "SVC": SVC(random_state = 32)
                }
            load_params = load_json("notebook/data/params.json")
            params = load_params["model_params"][self.modeling_type]

            model_report:dict=evaluate_models(X_train=X_train,
                                              y_train=y_train,
                                              X_test=X_test,
                                              y_test=y_test,
                                              models=models,
                                              param=params,
                                              modeling_type = self.modeling_type
                                              )
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.5:
                raise "No best model found"
            logging.info(f"Best found model on both training and testing dataset")

            default_model_file_path = self.model_trainer_config.trained_model_file_path
            basename_tag = self.model_trainer_config.basename_tag
            model_trainer_obj_file_path = default_model_file_path.replace(basename_tag, f"{self.modeling_type}_")
            save_object(
                file_path=model_trainer_obj_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            test_score = 0
            if self.modeling_type == "reg":
                test_score = r2_score(y_test, predicted)
            else:
                test_score = accuracy_score(y_test, predicted)
            return f"Best model: {best_model_name}\nBest model test score: {test_score}"
            
            
        except Exception as e:
            raise CustomException(e,sys)