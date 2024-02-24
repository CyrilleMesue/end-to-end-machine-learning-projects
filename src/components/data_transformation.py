import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler, LabelEncoder

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    basename_tag = "*"
    preprocessor_obj_file_path=os.path.join('artifacts',f"{basename_tag}preprocessor.pkl")

class DataTransformation:
    def __init__(self, 
                 modeling_type:str = "reg",
                 categorical_columns:list = [],
                 numerical_columns:list = [],
                 target_column_name:str = ""
                 ):
        """
        modeling_type can be either "clf" or "reg" for classification and regression repsectively.
        """
        self.data_transformation_config=DataTransformationConfig()
        self.modeling_type = modeling_type
        self.target_column_name = target_column_name
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        if target_column_name in self.categorical_columns:
            self.categorical_columns.remove(target_column_name)
        else:
            self.numerical_columns.remove(target_column_name)
            
            
    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {self.categorical_columns}")
            logging.info(f"Numerical columns: {self.numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,self.numerical_columns),
                ("cat_pipelines",cat_pipeline,self.categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            input_feature_train_df=train_df.drop(columns=[self.target_column_name],axis=1)
            target_feature_train_df=train_df[self.target_column_name]

            input_feature_test_df=test_df.drop(columns=[self.target_column_name],axis=1)
            target_feature_test_df=test_df[self.target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            if self.modeling_type == "clf":
                le = LabelEncoder()
                target_feature_train = le.fit_transform(target_feature_train_df.values)
                target_feature_test = le.transform(target_feature_test_df.values)
            else:
                target_feature_train = target_feature_train_df.values
                target_feature_test = target_feature_test_df.values

            train_arr = np.c_[
                input_feature_train_arr, target_feature_train
            ]
            test_arr = np.c_[input_feature_test_arr, target_feature_test]

            logging.info(f"Saved preprocessing object.")

            default_preprocessor_file_path = self.data_transformation_config.preprocessor_obj_file_path
            basename_tag = self.data_transformation_config.basename_tag
            preprocessing_obj_file_path = default_preprocessor_file_path.replace(basename_tag, f"{self.modeling_type}_")
            save_object(

                file_path=preprocessing_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)