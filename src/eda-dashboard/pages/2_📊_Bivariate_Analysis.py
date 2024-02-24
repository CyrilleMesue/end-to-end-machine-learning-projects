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
# Comparing Between two Categorical Variables
####################################################################################################
if "data" not in st.session_state:
    st.warning("Warning! No file uploaded. Please go to the home page, upload a file and **Run EDA**!")
else:
    data = st.session_state["data"]
    if len(st.session_state["categorical_features"]) >= 2:
        st.sidebar.caption("**Comparing Between 2 Categorical Features**")
        feature1 = st.sidebar.selectbox("Select 1st feature", st.session_state["categorical_features"])
        feature2 = st.sidebar.selectbox("Select 2nd feature", st.session_state["categorical_features"][::-1])
        st.caption(f"""
                **Comparing Between 2 Categorical Features**   
                1. What is the frequency or count of each combination of {feature1} and {feature2} variables?
                2. Are there any dominant combinations of categories between the {feature1} and {feature2} variables?
                3. How does the distribution of {feature1} variable vary within each category of the {feature2} variable?
                """)
        fig1,ax=plt.subplots(1,1,figsize=(20,8))
        sns.countplot(x=data[feature1],data=data,palette = 'bright',hue=feature2,saturation=0.95,ax=ax)
        ax.set_title(f'{feature1} vs {feature2}',color='#005ce6',size=20)
        for container in ax.containers:
            ax.bar_label(container,color='black',size=20)

        st.pyplot(fig1)
        ####################################################################################################
        # End
        ####################################################################################################
        

    ####################################################################################################
    # Comparing Between Categorical and Numeric Variables
    ####################################################################################################
    if len(st.session_state["categorical_features"]) >= 1 and len(st.session_state["numeric_features"]) >= 1:
        st.sidebar.caption("**Comparing Between a Categorical and Numeric Feature**")
        categorical_feature = st.sidebar.selectbox("Select categorical feature", st.session_state["categorical_features"])
        numeric_feature = st.sidebar.selectbox("Select numeric feature", st.session_state["numeric_features"])

        st.caption(f"""
                **Comparing Between a Categorical and Numeric Feature**
                1. Is the distribution of the "{numeric_feature}" variable symmetric, skewed to the left, or skewed to the right?
                2. Are there any clear peaks or modes in the distribution of the "{numeric_feature}" variable?
                3. Which "{categorical_feature}" category has the highest or lowest average value of the "{numeric_feature}" variable?
                4. How does the variability of the "{numeric_feature}" variable change across different "{categorical_feature}" categories?
                5. Are there any bimodal or multimodal patterns in the distribution of the "{numeric_feature}" variable?
    """)
        Group_data2=data.groupby(categorical_feature)
        fig2,ax=plt.subplots(1,2,figsize=(20,8))
        fig2.suptitle(f'{categorical_feature} vs {numeric_feature}',color='#005ce6',size=20)
        sns.barplot(x=Group_data2[numeric_feature].mean().index,y=Group_data2[numeric_feature].mean().values,palette = 'mako',ax=ax[0])

        for container in ax[0].containers:
            ax[0].bar_label(container,color='black',size=15)
        fig2.text(0.1, 0.5, f'{numeric_feature}', ha='center', va='center', rotation='vertical')

        sns.histplot(data=data,x=numeric_feature,kde=True,hue=categorical_feature, ax = ax[1])

        st.pyplot(fig2)
        ####################################################################################################
        # End
        ####################################################################################################