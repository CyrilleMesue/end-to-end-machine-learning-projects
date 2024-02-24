# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px

from src.exception import CustomException
from src.logger import logging
import os, sys

####################################################################################################
# Page settings
####################################################################################################
st.set_page_config(layout="wide")
st.title(':sparkles: Dynamic Exploratory Data Analysis!')
####################################################################################################
# End
####################################################################################################


####################################################################################################
# Data Transformation
####################################################################################################
class DataTransformation:
    def __init__(self, data, numerical_columns, categorical_columns):
        self.data = data
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns

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

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,self.numerical_columns),
                ("cat_pipelines",cat_pipeline,self.categorical_columns)

                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self):

        try:
            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            numerical_columns = ["writing_score", "math_score"]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            transformed=preprocessing_obj.fit_transform(data)

            return transformed
        except Exception as e:
            raise CustomException(e,sys)
####################################################################################################
# End
####################################################################################################
        
####################################################################################################
# PCA
####################################################################################################
if "data" not in st.session_state:
    st.warning("Warning! No file uploaded. Please go to the home page, upload a file and **Run EDA**!")
else:
    data = st.session_state["data"]

    if len(st.session_state["categorical_features"]) >= 1:
        hue = st.sidebar.selectbox("Select feature for hue", st.session_state["categorical_features"])
        target_col = st.sidebar.selectbox("Select target feature", st.session_state["categorical_features"])

        # transform data
        transformer = DataTransformation(data, st.session_state["numeric_features"], st.session_state["categorical_features"])
        transformed_data = transformer.initiate_data_transformation()
        scaled_data = transformed_data
        # scaled = StandardScaler()
        # scaled.fit(data)
        # scaled_data = scaled.transform(data)


        pca2 = PCA(n_components=2)
        pca2.fit(scaled_data)
        xpca2 = pca2.transform(scaled_data)

        pca3 = PCA(n_components=3)
        pca3.fit(scaled_data)
        xpca3 = pca3.transform(scaled_data)

        target_classes = data[target_col].unique()
        labels = data[target_col]

        # plot PCA 2
        # colors to use for visualization
        top_colors = [
            "salmon",
            "lightblue",
            "orange",
            "blue",
            "green",
            "navy",
            "gray",
            "red",
            "yellow",
        ]
        PC1 = 0
        PC2 = 1

        target_classes = data[hue].unique()
        colors = top_colors[: len(target_classes)]
        markers = ["o", "+", "*", "s", "x", "^"]
        axes = []
        case_study = "Student Result Prediction"


        figure1, ax = plt.subplots()
        figure1.set_size_inches(12, 6)
        xpca2_df = pd.DataFrame(xpca2, columns = ["PC1", "PC2"])
        xpca2_df[hue] = data[hue]

        sns.scatterplot(data=xpca2_df, x="PC1", y="PC2", hue=hue,style =hue, legend='auto');
        st.pyplot(figure1)

        pca_data = pd.concat([pd.DataFrame(xpca3), data[target_col]], axis=1)
        figure2 = px.scatter_3d(
            pca_data,
            x=0,
            y=1,
            z=2,
            color=target_col,
            labels={"0": "PC 1", "1": "PC 2", "2": "PC 3"},
        )

        figure2.update_yaxes(tickfont=dict(size=8))
        figure2.update_xaxes(tickfont=dict(size=12))
        figure2.update_layout(height=800, width = 1000)
        st.plotly_chart(figure2,height=800, width = 1000)

    else:
        # transform data
        transformer = DataTransformation(data, st.session_state["numeric_features"], st.session_state["categorical_features"])
        transformed_data = transformer.initiate_data_transformation()
        scaled_data = transformed_data
        # scaled = StandardScaler()
        # scaled.fit(data)
        # scaled_data = scaled.transform(data)


        pca2 = PCA(n_components=2)
        pca2.fit(scaled_data)
        xpca2 = pca2.transform(scaled_data)

        pca3 = PCA(n_components=3)
        pca3.fit(scaled_data)
        xpca3 = pca3.transform(scaled_data)

        # plot PCA 2
        PC1 = 0
        PC2 = 1

        markers = ["o", "+", "*", "s", "x", "^"]
        axes = []
      
        figure1, ax = plt.subplots()
        figure1.set_size_inches(12, 6)
        xpca2_df = pd.DataFrame(xpca2, columns = ["PC1", "PC2"])

        sns.scatterplot(data=xpca2_df, x="PC1", y="PC2", legend='auto');
        st.pyplot(figure1)

        pca_data = pd.DataFrame(xpca3)
        figure2 = px.scatter_3d(
            pca_data,
            x=0,
            y=1,
            z=2,
            labels={"0": "PC 1", "1": "PC 2", "2": "PC 3"},
        )

        figure2.update_yaxes(tickfont=dict(size=8))
        figure2.update_xaxes(tickfont=dict(size=12))
        figure2.update_layout(height=800, width = 1000)
        st.plotly_chart(figure2,height=800, width = 1000)
        ####################################################################################################
        # End
        ####################################################################################################