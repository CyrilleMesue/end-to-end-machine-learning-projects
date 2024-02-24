import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    basename_tag = "*"
    train_data_path: str=os.path.join('artifacts',f"{basename_tag}train.csv")
    test_data_path: str=os.path.join('artifacts',f"{basename_tag}test.csv")
    raw_data_path: str=os.path.join('artifacts',f"{basename_tag}data.csv")

class DataIngestion:
    def __init__(self, 
                 input_data_file_name:str = "stud.csv",
                 test_size:float = 0.2,
                 modeling_type:str = "reg",
                 target_column_name:str =""
                 ):
        
        self.ingestion_config=DataIngestionConfig()
        basename_tag = self.ingestion_config.basename_tag
        self.ingestion_config.train_data_path = self.ingestion_config.train_data_path.replace(basename_tag,f"{modeling_type}_")
        self.ingestion_config.test_data_path = self.ingestion_config.test_data_path.replace(basename_tag,f"{modeling_type}_")
        self.ingestion_config.raw_data_path = self.ingestion_config.raw_data_path.replace(basename_tag,f"{modeling_type}_")
        self.input_data_file_name = input_data_file_name
        self.test_size = test_size
        self.target_column_name = target_column_name
        self.modeling_type = modeling_type
        

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv(f'notebook/data/{self.input_data_file_name}')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            df = df[df[self.target_column_name].notna()].reset_index(drop=True) # remove entries where the target value is missing
            
            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=self.test_size,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            numerical_features = [feature for feature in df.columns if df[feature].dtype != 'O']
            categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']
            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                numerical_features,
                categorical_features

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    # simple user configurations
    modeling_type = "reg"
    target_column_name = "writing_score"
    input_data_file_name = "stud.csv"
    test_size = 0.4

    obj=DataIngestion(input_data_file_name = input_data_file_name, 
                      modeling_type = modeling_type,
                      test_size = test_size,
                      target_column_name = target_column_name
                      )
    train_data,test_data, numerical_columns, categorical_columns=obj.initiate_data_ingestion()

    data_transformation=DataTransformation(numerical_columns=numerical_columns,
                                           categorical_columns = categorical_columns,
                                           target_column_name = target_column_name,
                                           modeling_type = modeling_type
                                           )
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer(modeling_type = modeling_type)
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))