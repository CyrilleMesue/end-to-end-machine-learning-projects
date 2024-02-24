# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

####################################################################################################
# Page settings
####################################################################################################
st.set_page_config(layout="wide")
st.title(':sparkles: Dynamic Exploratory Data Analysis!')
####################################################################################################
# End
####################################################################################################


####################################################################################################
# Boxplot, Violin Plot, Barplot and Pieplot
####################################################################################################
if "data" not in st.session_state:
    st.warning("Warning! No file uploaded. Please go to the home page, upload a file and **Run EDA**!")
else:
    data = st.session_state["data"]
    if len(st.session_state["categorical_features"]) >= 1:
        st.sidebar.caption("**Barplot and PieChart**")
        selected_categorical_feature = st.sidebar.selectbox("Please select categorical feature", st.session_state["categorical_features"])

        st.caption(f"""
                **Barplot and PieChart for {selected_categorical_feature}**
                1. How is the distribution of "{selected_categorical_feature}" represented in the dataset?
                2. What is the percentage breakdown of groups within the "{selected_categorical_feature}" category?
                """)
        fig1,ax1=plt.subplots(1,2,figsize=(20,10))
        sns.countplot(x=data[selected_categorical_feature],data=data,palette = 'bright',ax=ax1[0],saturation=0.95)
        for container in ax1[0].containers:
            ax1[0].bar_label(container,color='black',size=20)
            
        explode = [0.1] + [0 for i in range(st.session_state["nunique_dict"][selected_categorical_feature]-1)]
        plt.pie(x = data[selected_categorical_feature].value_counts(),labels=data[selected_categorical_feature].value_counts().index,explode=explode,autopct='%1.1f%%',shadow=True)
        plt.show() 

        st.pyplot(fig1)
        ####################################################################################################
        # End
        ####################################################################################################

    ####################################################################################################
    # Boxplot and Violin Plots
    ####################################################################################################
    if len(st.session_state["numeric_features"]) >= 1:
        st.sidebar.caption("**Boxplot and Violin Plot**")
        selected_numeric_feature = st.sidebar.selectbox("Please select numeric feature", st.session_state["numeric_features"])

        st.caption(f"""
                **Boxplot and Violin Plot for {selected_numeric_feature}**
                1. Are there any outliers or extreme values in the dataset?
                2. Which Groups contain the largest and lowest number of observations as per the "{selected_numeric_feature}" category?
                3. How does the density of the data vary across different parts of the distribution according to the "{selected_numeric_feature}" category?
                4. Are there any multimodal distributions of the "{selected_numeric_feature}" category present in the data?
                """)
        
        fig2,ax2=plt.subplots(1,2,figsize=(20,10))
        sns.boxplot(data[selected_numeric_feature],color='skyblue', ax = ax2[0])
        sns.violinplot(y=selected_numeric_feature,data=data,color='red',linewidth=3, ax = ax2[1])

        st.pyplot(fig2)
        ####################################################################################################
        # End
        ####################################################################################################





