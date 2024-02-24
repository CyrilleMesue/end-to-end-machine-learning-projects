# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

####################################################################################################
# Page settings
####################################################################################################
st.set_page_config(layout="wide")
st.title(':sparkles: Dynamic Exploratory Data Analysis!')
####################################################################################################
# End
####################################################################################################

####################################################################################################
# Scatter Plots
####################################################################################################
if "data" not in st.session_state:
    st.warning("Warning! No file uploaded. Please go to the home page, upload a file and **Run EDA**!")
else:
    data = st.session_state["data"]

    if len(st.session_state["numeric_features"]) >= 1:
        st.sidebar.caption("**Comparing Between Multiple Numeric Features: Scatter Plots**")

        if len(st.session_state["categorical_features"]) >= 1:
            hue = st.sidebar.selectbox("Select hue", st.session_state["categorical_features"])
            selected_columns = st.sidebar.multiselect("Select features", st.session_state["numeric_features"], 
                                                        default = st.session_state["numeric_features"][:3]) + [hue]
            
            st.caption(f"""
                    **Comparing Between Multiple Numeric Features: Scatter Plots**   
                    1. What is the pairwise relationship between the selected features?
        """)
            fig1 = sns.pairplot(data.loc[:,selected_columns],hue = hue)
            st.pyplot(fig1)
        else:
            selected_columns = st.sidebar.multiselect("Select features", st.session_state["numeric_features"], 
                                                            default = st.session_state["numeric_features"][:3])
            
            st.caption(f"""
                    **Comparing Between Multiple Numeric Features: Scatter Plots**   
                    1. What is the pairwise relationship between the selected features?
        """)
            fig1 = sns.pairplot(data.loc[:,selected_columns])
            st.pyplot(fig1)
        ####################################################################################################
        # End
        ####################################################################################################

    ####################################################################################################
    # Comparing Between three Categorical Variables
    ####################################################################################################
    if len(st.session_state["categorical_features"]) >=3:
        st.sidebar.caption("**Comparing Between 3 Categorical Features: Countplots**")
        feature1 = st.sidebar.selectbox("Select 1st feature", st.session_state["categorical_features"], index =0)
        feature2 = st.sidebar.selectbox("Select 2nd feature", st.session_state["categorical_features"], index =1)
        feature3 = st.sidebar.selectbox("Select 3nd feature", st.session_state["categorical_features"], index =2)
        

        st.caption(f"""
                **Comparing Between 3 Categorical Features: Countplots**   
                1. Is there any cordial relationship between "{feature1}", "{feature2}" and "{feature3}"?
                2. Are there any dominant categories within each variable, and how do they compare across the three variables?
    """)
        fig2,ax2=plt.subplots(1,2,figsize=(20,8))
        sns.countplot(x=data[feature1],data=data,palette = 'bright',hue=feature2,saturation=0.95,ax=ax2[0])
        ax2[0].set_title(f'{feature2}',color='#36454F',size=20)
        ax2[1].set_title(f'{feature3}',color='#36454F',size=20)
        for container in ax2[0].containers:
            ax2[0].bar_label(container,color='#36454F',size=20)
            
        sns.countplot(x=data[feature1],data=data,palette = 'bright',hue=feature3,saturation=0.95,ax=ax2[1])
        for container in ax2[1].containers:
            ax2[1].bar_label(container,color='#36454F',size=20)  

        fig2.suptitle(f'{feature1} vs {feature2} vs {feature3}',color='#005ce6',size=20)
        st.pyplot(fig2)
        ####################################################################################################
        # End
        ####################################################################################################

    ####################################################################################################
    # Histogram and KDE Plots: Comparing Between two Categorical and one Numeric Variables
    ####################################################################################################
    if len(st.session_state["categorical_features"]) >= 2 and len(st.session_state["numeric_features"]) >= 1:
        st.sidebar.caption("**Comparing Between 2 Categorical and 1 Numeric Feature(s): Histogram and KDE Plots**")
        numeric_feature = st.sidebar.selectbox("Select numeric feature", st.session_state["numeric_features"])
        categorical_feature1 = st.sidebar.selectbox("Select 1st categorical feature", st.session_state["categorical_features"])
        categorical_feature2 = st.sidebar.selectbox("Select 2nd categorical feature", st.session_state["categorical_features"][::-1])

        st.caption(f"""
                **Comparing Between 2 Categorical and 1 Numeric Feature(s): Histogram and KDE Plots**
                1. How does the variability of the data differ across different groups or categories?
                2. Are there any notable trends or patterns in the relationship between "{categorical_feature2}" and "{numeric_feature}"?
                3. How does the distribution of "{numeric_feature}" vary across different "{categorical_feature1}" and "{categorical_feature2}" categories?
                
    """)

        # get number of plots
        unique_classes = data[categorical_feature2].unique()
        n_unique_classes = data[categorical_feature2].nunique()
        num_plots = 1 + n_unique_classes
        num_cols = 2
        num_rows = math.ceil(num_plots / num_cols)


        fig3,ax3 = plt.subplots(num_rows,num_cols,figsize=(20,8*num_rows))
        fig3.suptitle(f'{categorical_feature1} vs {categorical_feature2} vs {numeric_feature}',color='#005ce6',size=20)

        def pattern(n):
            output = [0,0]
            for i in range(n-2):
                output = output + [output[-2]+1, output[-1]+1]
            return output[:n]
        col1 = pattern(num_plots)
        col2 = list(map(lambda x: int(0.5-0.5*(-1)**x), range(num_plots)))

        sns.histplot(data=data,x=numeric_feature,kde=True,hue=categorical_feature1, ax = ax3[0,0])
        ax3[0,0].set_title(f'All data',color='#36454F',size=20)
        for i in range(1,num_plots):
            unique_class = unique_classes[i-1]
            sns.histplot(data=data[data[categorical_feature2]==unique_class],x=numeric_feature,kde=True,hue=categorical_feature1, ax = ax3[col1[i], col2[i]])
            ax3[col1[i], col2[i]].set_title(f'{categorical_feature2}: {unique_classes[i-1]}',color='#36454F',size=20)
        st.pyplot(fig3)
        ####################################################################################################
        # End
        ####################################################################################################