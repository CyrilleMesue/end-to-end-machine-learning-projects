import os
import sys

import numpy as np 
import pandas as pd
import json
import dill
import pickle
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param, modeling_type = "reg"):
    try:
        report = {}

        if modeling_type =="reg":
            kfold = KFold(n_splits=5, random_state=32, shuffle = True)
            scoring = "r2"
            score_test = lambda x,y : r2_score(x,y)
        else:
            kfold = StratifiedKFold(n_splits=5, random_state=32, shuffle = True)
            scoring = "accuracy"
            score_test = lambda x,y : accuracy_score(x,y)

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=kfold, scoring = scoring)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = score_test(y_train, y_train_pred)

            test_model_score = score_test(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def load_json(json_path):
    """
    Load json data as a dictionary
    """

    try:
        with open(json_path) as f:
            file = f.read()
        json_data = json.loads(file)

    except Exception as e:
        raise CustomException(e, sys)

    return json_data