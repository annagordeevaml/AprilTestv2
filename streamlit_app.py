# IMPORTING PACKAGES
# region
import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import pygsheets
import matplotlib.pyplot as plt
from gspread_dataframe import set_with_dataframe
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import snowflake.connector
import warnings;
warnings.simplefilter('ignore')
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
import os
from snowflake.connector.pandas_tools import write_pandas, pd_writer
from datetime import datetime, timedelta
from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tools.eval_measures import rmse
import holidays
from prophet import Prophet
import snowflake.connector
import warnings;
warnings.simplefilter('ignore')
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
import os
from snowflake.connector.pandas_tools import write_pandas, pd_writer
import gspread
from google.oauth2.service_account import Credentials
from hashlib import sha256
# endregion

# FUNCTIONS
#region
# Define a function for clean data formatting for display
def format_dataframe(df):
    return df.style.format("{:,.0f}").set_properties(**{
        'text-align': 'right',
        'white-space': 'nowrap'
    }).set_table_styles([{
        'selector': 'th',
        'props': [('text-align', 'left')]
    }])

# Function to remove special characters and convert to float
def clean_and_convert(value):
    try:
        return float(value.replace(',', '').replace('$', '').replace('%', ''))
    except ValueError:
        return value

# Define the color styling functions with paler colors
def color_pacing(value):
    if isinstance(value, float) or isinstance(value, int):
        return 'background-color: #e6ffe6' if value > 0 else 'background-color: #ffe6e6'  # Pale green and pale red
    return ''

def color_attainment(value):
    if isinstance(value, float) or isinstance(value, int):
        return 'background-color: #e6ffe6' if value >= 100 else 'background-color: #ffe6e6'  # Pale green and pale red
    return ''

def color_actuals_last_year_text(value):
    return 'color: #d3d3d3'  # Light grey text color
#endregion

# Authentication with Google Sheets
# region
scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets',
         "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
gc = pygsheets.authorize(service_file='/Users/anna.gordeeva/Desktop/SSS/autocred.json')
creds = ServiceAccountCredentials.from_json_keyfile_name('/Users/anna.gordeeva/Desktop/SSS/autocred.json', scope)
client = gspread.authorize(creds)
# endregion

# Google Sheet URLs
# region

# MVP_FORECAST GOOGLE SHEET URL
mvp_forecast_url = 'https://docs.google.com/spreadsheets/d/1d51YMLHSkta3RlmD8L9pJ2sNMCH6n2KznbGsbsqQJH8/edit#gid=0'

# MVP_FORECAST: Graphs for Total ARR
graphs_analysis_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQbWSrVI_Mrcg-8uLNvjavLRCs932aX4vQUsFGCIj_PleCRhrhIhzZKYCsu6yFa2LAr4sPoMZ6OhTl4/pubchart?oid=2017778288&format=interactive"

# Graph:  MVP_FORECAST -> Analysis -> Weekly Tracking
mvp_forecast_analysis_weeklytracker_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQbWSrVI_Mrcg-8uLNvjavLRCs932aX4vQUsFGCIj_PleCRhrhIhzZKYCsu6yFa2LAr4sPoMZ6OhTl4/pubchart?oid=1840769526&format=interactive"

# Graph:  MVP_FORECAST -> Analysis -> Total Weekly Tracking
mvp_forecast_analysis_weeklytracker_totaltracker = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQbWSrVI_Mrcg-8uLNvjavLRCs932aX4vQUsFGCIj_PleCRhrhIhzZKYCsu6yFa2LAr4sPoMZ6OhTl4/pubchart?oid=2095833483&format=interactive"
# endregion

# CONNECTING TO GOOGLE SHEETS TABLES AND GRAPHS
# region

# Access the sheet for MVP_FORECST.Actuals
sheet = client.open_by_url(mvp_forecast_url)
worksheet = sheet.worksheet('Actuals')
data = worksheet.get_all_values()
actuals = pd.DataFrame(data)
actuals.columns = actuals.iloc[0]  # Set first row as header
actuals = actuals.drop(actuals.index[0])


# Access the sheet for MVP_FORECST.Analysis
worksheet_analysis = sheet.worksheet('Analysis')
table_analysis = worksheet_analysis.get('A4:H8')
table_analysis = pd.DataFrame(table_analysis)
table_analysis.columns = table_analysis.iloc[0]
table_analysis = table_analysis.drop(0)
table_analysis = table_analysis.reset_index(drop=True)

# MVP_FORECST: Different tabs
sheet_mvp_forecast_rawdata = sheet.worksheet('Raw Data')
sheet_mvp_forecast_analysis = sheet.worksheet('Analysis')
sheet_mvp_forecast_test = sheet.worksheet('Test')
sheet_mvp_forecast_actuals = sheet.worksheet('Actuals')
sheet_mvp_forecast_forecast = sheet.worksheet('Forecast')
sheet_mvp_forecast_forecastmelted = sheet.worksheet('Forecast_melted')
sheet_mvp_forecast_kaidforecast = sheet.worksheet('KAID Forecast')

# Info Table for tracking Dates
worksheet_analysis = sheet.worksheet('Analysis')
table_analysis_infotable = worksheet_analysis.get('A1:D2')
table_analysis_infotable = pd.DataFrame(table_analysis_infotable)
table_analysis_infotable.columns = table_analysis_infotable.iloc[0]
table_analysis_infotable = table_analysis_infotable.drop(0)
table_analysis_infotable = table_analysis_infotable.reset_index(drop=True)



# endregion

# TABLES STYLES
# region

# Total: Pacing YoY and % Attainment style
table_analysis['Pacing YoY'] = table_analysis['Pacing YoY'].apply(clean_and_convert)
table_analysis['Pacing YoY'] = table_analysis['Pacing YoY'].round(2)
table_analysis['% Attainment'] = table_analysis['% Attainment'].apply(clean_and_convert)

styled_table = table_analysis.style.applymap(color_pacing, subset=['Pacing YoY'])
styled_table = styled_table.applymap(color_attainment, subset=['% Attainment'])
styled_table = styled_table.format({'Pacing YoY': '{:.1f}%', '% Attainment': '{:.1f}%'})
styled_table = styled_table.applymap(color_actuals_last_year_text, subset=['Actuals Last Year'])
styled_table = styled_table.applymap(color_actuals_last_year_text, subset=['Actuals'])

# endregion

# STREAMLIT PASSWORD
# region
def set_custom_width():
    st.markdown(
        f'''
        <style>
            .reportview-container .main .block-container{{
                max-width: 95%;
            }}
            .reportview-container .main {{
                color: {st.config.get_option('theme.primaryColor')};
                background-color: {st.config.get_option('theme.backgroundColor')};
            }}
        </style>
        ''',
        unsafe_allow_html=True,
    )

st.set_page_config(page_title='Forecast App', layout='wide')
set_custom_width()


# Функция для проверки пароля
def check_password():
    # Хэш реального пароля (предполагается, что вы заранее сгенерировали хэш вашего пароля)
    correct_password_hash = "AprilTest"

    # Виджет ввода пароля
    password = st.sidebar.text_input("Password:", value="", type="password")

    # Проверяем пароль
    if password:
        if password == correct_password_hash:
            return True
        else:
            st.sidebar.error("Incorrect Password")
            return False
    else:
        return False

# endregion

# Streamlit page configuration
if check_password():

    selected_month = st.selectbox('Select a Month', ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    sheet_mvp_forecast_analysis.update('A37', selected_month)

    current_year = table_analysis_infotable[ 'Year'].iloc[0]  # This gets the first item in the 'Current Year' column
    current_month = table_analysis_infotable['Month'].iloc[0]
    current_day = table_analysis_infotable['Days'].iloc[0]
    days_in_current_month = table_analysis_infotable['Days in chosen month'].iloc[0]

    # Last Forecast Update
    # region
    def update_click_time():
        st.session_state['last_clicked'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if 'last_clicked' not in st.session_state:
        st.session_state['last_clicked'] = 'Never'



    # endregion

    # FORECAST COOKING
    # region

    st.sidebar.header('Cook a new Forecast')
    option = st.sidebar.selectbox(
        'Do you want to cook a new forecast?',
        ('No', 'Yes')
    )

    if option == 'Yes':
        st.session_state['last_clicked'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        st.sidebar.write('Last Forecast Update')
        st.sidebar.write(st.session_state['last_clicked'])
    else:
        st.sidebar.write(st.session_state['last_clicked'])

    if option == 'Yes':
        try:

            con = snowflake.connector.connect(
                user = 'ANNA.GORDEEVA@KLAVIYO.COM',
                password = 'IlonAndRaf_23',
                account='klaviyo.snowflakecomputing.com',
                database = 'KLAVIYO',
                schema = 'sandbox',
                warehouse = 'MARKETING_OPS_XSMALL',
                region= 'us-east-1'
            )

            select_query = f'''
            select Account_Created_Date, region , ecommerce_platform, count(distinct klaviyo_account_id) as Accounts
            from(
            select date_trunc('day', klaviyo_account_created_date)::date as Account_Created_Date
                , coalesce(won_aggregated_location,opp_aggregated_location,initial_aggregated_location) as region
                , case when coalesce(won_aggregated_ecommerce_platform,opp_aggregated_ecommerce_platform,initial_aggregated_ecommerce_platform) in ('Shopify','Shopify Plus')             then 'Shopify' 
                        else 'Non-Shopify' 
                    end as ecommerce_platform
                    , klaviyo_account_id as klaviyo_account_id
                , count(distinct klaviyo_account_id) as Accounts
                , sum(opportunity_line_item_amount) as mrr
            from  klaviyo.public.tbl_acquisition_funnel 
            where klaviyo_account_created_date >= '2018-01-01'
            and bad_actor_flag != 'Bad Actor Lead'
            group by 1,2,3,4
            order by 1,2,3
            )
            group by 1,2,3
            order by 1,2,3
            
            '''

            cursor = con.cursor()
            cursor.execute(select_query)
            rows = cursor.fetchall()

            df = pd.DataFrame(rows)
            df.columns = ['Account_Created_Date', 'region', 'ecommerce_platform', 'Accounts']
            grouped = df.groupby(['Account_Created_Date', 'region', 'ecommerce_platform'])['Accounts'].sum().reset_index()
            pivot_table = grouped.pivot(index=[ 'region', 'ecommerce_platform'],columns='Account_Created_Date' , values='Accounts').reset_index()
            pivot_table = pivot_table.fillna(0)

            daysforforecast = 365


            def flipped_df(pv_12):
                date_columns = [col for col in pv_12.columns if '-' in str(col)]
                result_df = pv_12[date_columns]
                flipped_df = result_df.T
                flipped_df = flipped_df.reset_index()
                flipped_df = flipped_df.rename(columns={'Account_Created_Date': 'ds', pv_12.index[0]: 'y'})
                return flipped_df
            def prophet(table, x):
                model = Prophet(interval_width=0.95, daily_seasonality=True)
                model.fit(table)
                future = model.make_future_dataframe(periods=x, freq='D')
                forecast = model.predict(future)
                return forecast
            datasets = {}
            for index, row in pivot_table.iterrows():
                dataset_name = index
                new_dataset = pd.DataFrame([row])
                datasets[dataset_name] = new_dataset
            dataresults = {}
            for i in range(0, len(datasets)):
                table = flipped_df(datasets[i])
                dataresults[i] = prophet(table, daysforforecast)
            dataconcat = []
            for i in range(len(datasets)):
                dataconcat.append(datasets[i])  # Append each DataFrame to the list
            concatenated_df = pd.concat(dataconcat, axis=0)
            from datetime import datetime, timedelta
            colnames = []
            for i in range(1, daysforforecast+1):
                date_datetime = datetime.strptime(str(pivot_table.columns[-1]), '%Y-%m-%d')
                date_datetime = date_datetime  + timedelta(days=i)
                date_string = date_datetime.strftime('%Y-%m-%d')
                colnames.append(date_string)
            finres = []
            for i in range(len(pivot_table)):
                k = dataresults[i]['yhat'].tail(daysforforecast).abs()
                finres.append(pd.DataFrame(k).transpose())  # Append each DataFrame to the list

            concatenated_finres = pd.concat(finres, axis=0)
            concatenated_finres.reset_index(drop=True, inplace=True)
            concatenated_finres.columns = colnames

            pivot_table.reset_index(drop=True, inplace=True)
            resultingforecast = pd.concat([pivot_table, concatenated_finres], axis=1)

            reshaped_data = pd.melt(resultingforecast, id_vars=[ 'region', 'ecommerce_platform'],
                                   var_name='Account_Created_Date', value_name='Accounts')
            reshaped_data.columns = ['region', 'ecommerce_platform', 'Account_Created_Date', 'Accounts']
            reshaped_data["Account_Created_Date"] = pd.to_datetime(reshaped_data["Account_Created_Date"])
            reshaped_data["Account_Created_Date"] = pd.to_datetime(reshaped_data["Account_Created_Date"]).dt.strftime('%Y-%m-%d %H:%M:%S')
            reshaped_data.insert(0, 'Count', range(1, len(reshaped_data) + 1))

            #url_total = 'https://docs.google.com/spreadsheets/d/1LY8J8-IBREn_EdUkpYToJIXiGpQ6u4pp7WMH4m_c4fc/edit?usp=sharing'
            #sheet_total = client.open_by_url(url_total)
            #worksheet_total = sheet_total.worksheet('3. Accounts Forecast')
            #worksheet_total.clear()
            #set_with_dataframe(worksheet_total, reshaped_data)

            sheet = client.open_by_url(mvp_forecast_url)
            worksheet = sheet.worksheet('KAID Forecast')
            worksheet.clear()
            set_with_dataframe(worksheet, reshaped_data)

            # NEW FORCAST MELTING
            # region
            sheet = client.open_by_url(mvp_forecast_url)
            worksheet_forecast = sheet.worksheet('Forecast')
            data_forecast = worksheet_forecast.get_all_values()
            forecast = pd.DataFrame(data_forecast)
            forecast.columns = forecast.iloc[0]  # Set first row as header
            forecast = forecast.drop(forecast.index[0])
            for i in range(1, 13):
                forecast[str(i)] = forecast[str(i)].replace('[\$,]', '', regex=True).astype(float)
            forecast_melted = forecast.melt(
                id_vars=['Geo', 'New Segment', 'Product', 'New vs Base vs NA', 'Shop/Nonshop', 'Sales Type'],
                var_name='Month', value_name='ARR')
            sheet.worksheet('Forecast_melted').clear()
            set_with_dataframe(sheet.worksheet('Forecast_melted'), forecast_melted)
            # endregion

        except Exception as e:
            st.error(f'Connection Error: {e}')

    else:
        # Если пользователь выбрал что-то другое, код соединения не выполняется
        st.write("")

    # endregion


    # Streamlit Application

    # Title
    st.title('Forecast Tracker')

    # Sidebar for user inputs
    st.sidebar.header('Current Date')
    st.sidebar.text(f'Year: {current_year}')
    st.sidebar.text(f'Month: {current_month}')
    st.sidebar.text(f'Day: {current_day}')

    st.sidebar.header('User Input Features')
    conversion_rate = st.sidebar.text_input('Enter a custom conversion rate:', '0.02')

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Total", "Americas", "EMEA", "APAC"])

    # ARR Data Table
    with tab1:
        st.dataframe(styled_table)
         # Graphs for ARR
        st.markdown(f'<iframe src="{graphs_analysis_url}" width="1200" height="400"></iframe>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            # Graphs for weekly tracking
            st.markdown(f'<iframe src="{mvp_forecast_analysis_weeklytracker_url}" width="700" height="1800"></iframe>', unsafe_allow_html=True)

        with col2:
            # Graphs for weekly tracking
            st.markdown(f'<iframe src="{mvp_forecast_analysis_weeklytracker_totaltracker}" width="700" height="1800"></iframe>', unsafe_allow_html=True)


