# import libraries
import streamlit as st
import pandas as pd
import numpy as np

####################################################################################################
# Page settings
####################################################################################################
st.set_page_config(layout="wide")
st.title(':sparkles: Dynamic Exploratory Data Analysis!')

# set sidebar width
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 360px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 360px;
        margin-left: -500px;
    }
    </style>
    """,
    unsafe_allow_html=True,

)

# fix margins
st.markdown("""
    <style>
        .main > div {
            padding-left: 3rem;
            padding-right: 3rem;
        }
    </style>
""", unsafe_allow_html=True)
####################################################################################################
# End
####################################################################################################



####################################################################################################
# Load Data and Process Data
####################################################################################################
# create 2 column layouts
part0col1, part0col2 = st.columns([8,4])
part0col1.markdown("### Upload Data and Run EDA")  

# Option 1 : Load data from a file in either csv, smi or text format
uploaded_file = part0col1.file_uploader("Please upload data in csv format", accept_multiple_files=False, type = ['csv'])

load_data = part0col1.button("Run EDA", key = 1)
load_sample_data = part0col1.button("Run Sample Test", key = 2)

if load_data:
    if uploaded_file == None:
        part0col1.warning("Warning! No file uploaded.")
        
    else:
        # read uploaded file
        uploaded_data = pd.read_csv(uploaded_file,header=0,encoding='utf-8')
        
        st.session_state["raw_data"] = uploaded_data
        # tell the user that file has been loaded successfully
        part0col1.success("File content has been successfully accessed!!!")

# run on sample dataset
if load_sample_data:
    # read uploaded file
    file_path = "https://raw.githubusercontent.com/CyrilleMesue/Heart-Failure-Prediction/main/heart.csv"
    uploaded_data = pd.read_csv(file_path,header=0,encoding='utf-8')
    
    st.session_state["raw_data"] = uploaded_data
    # tell the user that file has been loaded successfully
    part0col1.success("Sample file content has been successfully accessed!!!")

if "raw_data" not in st.session_state:
    st.warning("Warning! No file uploaded. Please upload a file and **Run EDA**!")
else:
    # load data
    data = st.session_state["raw_data"].copy()
    st.write(st.session_state["raw_data"])
    # display data
    st.caption("**Dsiplay Dataset**")
    container1 = st.container()
    part1col1, part1col2 = container1.columns([8,4])
    part1col1.dataframe(data)
    part1col2.info("""
    **INFO:** 
    The dataset is presented on the left. Please scroll through the rows and columns to ensure that you have uploaded the correct dataset.

    Explore the summary sections below to gain insight into your dataset. Look for missing values, duplicates, and the features included in your dataset. Determine the number of numeric and categorical features, and examine summary statistics for numeric variables. You can also opt to utilize only a sample of the dataset for visualization purposes.    
    """) 

    # compute mean column
    numeric_features = [feature for feature in data.columns if data[feature].dtype != 'O']
    mean_columns = st.multiselect("Compute Mean Over Columns", numeric_features, default=numeric_features[:2])
    mean_values = data.loc[:,mean_columns].mean(axis =1)
    data.insert(0,"average",mean_values)

    # checking duplicates
    col1, col2, col3 = st.columns([4,5,3])
    col1.caption(f"**Number of duplicates:  {data.duplicated().sum()}**")
    remove_duplicates = col1.checkbox("Remove Duplicates?")

    if remove_duplicates:
        data = data.drop_duplicates().reset_index(drop=True)

    # select ordinal numeric features and convert to categorical
    ordinal_numeric_features = col2.multiselect("Select any columns containing ordinal numeric values", data.columns)
    convert_dict = {feature: "O" for feature in ordinal_numeric_features}
    data = data.astype(convert_dict) 

    # sample data
    sample_options  = {"100%": 1.0, "75%" : 0.75, "50%" : 0.50, "25%" : 0.25}
    display_options = sorted(list(sample_options.keys()))
    sample_amount = col3.selectbox("Sample data by:", sample_options.keys(), index = 3)
    data = data.sample(int(sample_options[sample_amount]*len(data)))
    data = data.reset_index(drop = True)

    # chech data summary
    datatype = data.dtypes
    non_null_count = data.count()
    null_count = data.isna().sum()
    unique_count = data.nunique()
    summary_df = pd.DataFrame({"Features": data.columns.values,
                            "Data Types": datatype.astype("string"),
                            "Number of Non-null Values": non_null_count,
                            "Number of Null Values": null_count,
                            "Number of Unique Values": unique_count
                            })
    st.caption("**Display Data Info**")
    container2 = st.container()
    part2col1, part2col2 = container2.columns([8,4])
    part2col1.write(summary_df)
    part2col2.info("""**Questions?**""") 
    part2col2.info("""
    1. How many missing values are there in each column?
    2. Which columns have missing values?
    3. Are there any duplicate rows in the dataset? How many?
    4. How many unique values does each column contain?,
    5. Do the data types align with the expected types?
    """) 

    # Check statistics of data set
    st.write("\n\n")
    st.caption("**Display Summary Statistics**")
    container3 = st.container()
    part3col1, part3col2 = container3.columns([8,4])
    part3col1.write(data.describe())
    part3col2.info("""**Questions?**""") 
    part3col2.info("""
    1. What are the summary statistics for numeric variables (mean, median, standard deviation, min, max, quartiles)?
    2. Do any statistics stand out as unusual or unexpected?
    """) 

    # Get Categorical And Numeric Features
    st.session_state["numeric_features"] = [feature for feature in data.columns if data[feature].dtype != 'O']
    st.session_state["categorical_features"] = [feature for feature in data.columns if data[feature].dtype == 'O']

    # store data and other variables in session state
    st.session_state["data"] = data
    st.session_state["nunique_dict"] = data.nunique().to_dict()
    ####################################################################################################
    # End
    ####################################################################################################
