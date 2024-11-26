# Importing ToolKits
import re
from time import sleep
import pandas as pd
import numpy as np
import streamlit as st
from streamlit.components.v1 import html
import warnings
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from model import *
import pickle


def run():
    st.set_page_config(
        page_title="Risk Level Detection",
        page_icon="👷",
        layout="wide"  
    )

    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Function To Load Our Dataset
    @st.cache_data
    
    def load_model(model_path):
        return pd.read_pickle(model_path)

    st.markdown(
        """
 <style>
    .main {
        text-align: center;
        background: linear-gradient(135deg, #1c1c1c, #2e2e2e);
        padding: 2rem;
        color: #e0e0e0;
    }

    h1 {
        font: bold 32px Arial, sans-serif;
        text-align: center;
        color: #f5f5f5;
        margin-bottom: 1rem;
        text-shadow: 1px 1px 4px rgba(0, 0, 0, 0.3);
    }

    h3 {
        font-size: 22px;
        color: #dcdcdc;
        margin-top: 1rem;
    }

    div[data-testid=stSidebarContent] {
        background-color: #202020;
        border-right: 4px solid #333;
        padding: 16px!important;
    }

    .block-container {
        padding-top: 1rem;
        background-color: #2b2b2b;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        padding: 2rem;
        color: #ffffff; /* Makes all text inside the block white */
    }

    .plot-container.plotly {
        border: 1px solid #555;
        border-radius: 6px;
        margin: 1rem 0;
    }

    .st-emotion-cache-16txtl3 {
        padding: 2rem 1rem;
        border-radius: 12px;
        background: #3a3a3a;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        color: #ffffff;
    }

    .st-emotion-cache-1r6slb0 {
        font: bold 24px Tahoma;
        color: #e0e0e0;
    }

    .st-emotion-cache-z5fcl4 {
        padding: 1.5rem;
        overflow-x: hidden;
        background: #2e2e2e;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        color: #ffffff;
    }

    /* Specific styling for labels */
    label {
        color: #ffffff; /* Ensures all labels are white */
    }

    div[data-baseweb=select] > div {
        cursor: pointer;
        background-color: #3a3a3a;
        border: 2px solid #444;
        border-radius: 8px;
        padding: 8px;
        transition: border 0.2s ease;
        color: #e0e0e0;
    }
    div[data-baseweb=select] > div:hover {
        border: 2px solid #5a9bd6;
    }

    div[data-baseweb=base-input] {
        background-color: #3a3a3a;
        border: 2px solid #444;
        border-radius: 8px;
        padding: 8px;
        transition: border 0.2s ease;
        color: #e0e0e0;
    }

    div[data-testid=stFormSubmitButton] > button {
        width: 50%;
        background-color: #5a9bd6;
        border: 2px solid #4b89c5;
        padding: 16px;
        font-size: 18px;
        font-weight: bold;
        color: #ffffff;
        border-radius: 30px;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    div[data-testid=stFormSubmitButton] > button:hover {
        background-color: #4b89c5;
        border-color: #3c6b9e;
    }

    input[type=number] {
        width: 100% !important;
        padding: 8px;
        border-radius: 6px;
        box-sizing: border-box;
        border: 1px solid #444;
        background-color: #3a3a3a;
        color: #e0e0e0;
        transition: border 0.2s ease;
    }
    input[type=number]:focus {
        border-color: #5a9bd6;
    }

    /* Adjust labels and titles */
    .st-emotion-cache-z5fcl4 h3, .st-emotion-cache-16txtl3 h1 {
        color: #dcdcdc;
    }
</style>



    """,
        unsafe_allow_html=True
    )

    header = st.container()
    content = st.container()

    st.write("")

    with header:
        st.title("Safety Intelligence - Body Injury prediction 👷")
        st.write("")

    with content:
        col1, col2 = st.columns([10, 5])

        with col1:
            with st.form("Preidct"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    env_factor = st.selectbox('Environmental Factor', options=['Catch Point/Puncture Action', 'Pinch Point Action', 'Materials Handling Equip./Method','Work-Surface/Facility-Layout Condition','Overhead Moving/Falling Object Action','Temperature +/- Tolerance Lev.', 'Sound Level','Chemical Action/Reaction Expos', 'Flying Object Action','Flammable Liquid/Solid Exposure', 'Overpressure/Underpressure','Weather, Earthquake, Etc.', 'Gas/Vapor/Mist/Fume/Smoke/Dust','Shear Point Action', 'Illumination', 'Radiation Condition'], index=0)

                    # max_heart_rate = st.number_input('Max Heart Rate', min_value=0,
                    #                                  max_value=200, value=100)
                    temp = st.number_input('Temperature', min_value=0, max_value=100, value=0)
                    dewpt = st.number_input('Dew Point', min_value=0, max_value=100, value=0)

                with c2:
                    task_assigned = st.selectbox('Task Regularity', options=[
                        "Regularly Assigned", "Not Regularly Assigned"], index=0)
                    feels = st.number_input('Feels Like', min_value=0, max_value=100, value=0)
                    heatind = st.number_input('Heat Indication', min_value=0, max_value=100, value=0)


                with c3:
                    arbitary = st.selectbox('Human Factors', options=[
                        "Regularly Assigned", "Not Regularly Assigned"], index=0)
                    wchills = st.number_input('Wind chills', min_value=0, max_value=100, value=0)
                    precips = st.number_input('Precipitation', min_value=0, max_value=100, value=0)

                predict_button = st.form_submit_button("Predict")

        with col2:
            result = ""
            if predict_button:
                variables = [env_factor, temp, dewpt, task_assigned, feels, heatind, arbitary, wchills, precips]
                new_data = pd.DataFrame({
                        'Task Assigned': [1],
                        'Temp': [float(temp)],
                        'Dps': [float(dewpt)],
                        'FeelsLike': [float(feels)],
                        'Heatind': [float(heatind)],
                        'Wchills': [float(wchills)],
                        'Precips': [float(precips)],
                        'Environmental Factor': [env_factor]
                    })
                
                with open('injury_prediction_model1.pkl', 'rb') as f:
                    loaded_model = pickle.load(f)
                # Initialize the class
                result = loaded_model.svmres(new_data)

            # Use Streamlit components to display the result
            with st.spinner(text='Predicting the Result...'):
                sleep(1.2)  # Optional: add delay for UX

                # Display the predicted body part
                # st.image("imgs/body_part.png", caption="", width=100) 
                st.subheader("Predicted Body Part:")
                st.subheader(f":blue[{result}]")  # Display the body part result in blue

                    # with heart_disease:
                    #     st.image("imgs/heart.png", caption="", width=65)
                    #     st.subheader(":green[*Not Heart Patient*]")
                    #     st.subheader(f"{prediction_prop[0, 0]}%")

                    # with no_heart_disease:
                    #     st.image("imgs/hearted.png", caption="", width=65)
                    #     st.subheader(f":red[*Heart Patient*]")
                    #     st.subheader(f"{prediction_prop[0, 1]}%")

run()
