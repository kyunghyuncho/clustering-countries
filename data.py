import pandas as pd
import numpy as np
import wbgapi as wb
import streamlit as st
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

INDICATORS = {
    'NY.GDP.PCAP.CD': 'GDP per capita',
    'NY.GDP.MKTP.KD.ZG': 'GDP growth',
    'SP.DYN.LE00.IN': 'Life expectancy',
    'EN.POP.DNST': 'Population density',
    'SP.POP.GROW': 'Population growth',
    'SL.UEM.TOTL.ZS': 'Unemployment',
    'FP.CPI.TOTL.ZG': 'Inflation',
    'IT.NET.USER.ZS': 'Internet users (%)',
    'SI.POV.GINI': 'Gini index (Inequality)',
    'MS.MIL.XPND.GD.ZS': 'Military exp. (% of GDP)',
    'VC.IHR.PSRC.P5': 'Homicide rate (per 100k)',
    'SG.GEN.PARL.ZS': 'Women in parliament (%)',
    'EG.USE.PCAP.KG.OE': 'Energy use per capita'
}

SKEWED_FEATURES = ['GDP per capita', 'Population density', 'Gini index (Inequality)', 'Homicide rate (per 100k)', 'Energy use per capita']

@st.cache_data(show_spinner="Fetching indicator data from World Bank API...")
def load_data():
    """
    Fetches the latest available indicator data using wbgapi and format as dataframe.
    """
    indicators_list = list(INDICATORS.keys())
    
    # Fetch the last 10 years of data to maximize coverage
    df = wb.data.DataFrame(indicators_list, mrv=10)
    
    # Backfill missing values along the rows, then select the most recent (first) column
    recent_values = df.bfill(axis=1).iloc[:, 0]
    recent_values.name = 'value'
    
    df_recent = recent_values.reset_index()

    # Get iso3 country codes and country names to filter out aggregates (regions, etc.)
    economies = wb.economy.DataFrame()
    countries = economies[economies['aggregate'] == False]
    country_codes = countries.index.tolist()
    
    df_recent = df_recent[df_recent['economy'].isin(country_codes)]
    
    # Pivot to make rows countries and columns indicators
    df_pivoted = df_recent.pivot(index='economy', columns='series', values='value')
    # Rename columns to human readable names
    df_pivoted.rename(columns=INDICATORS, inplace=True)
    
    # Merge country names into the dataframe
    df_pivoted = df_pivoted.merge(countries[['name']], left_index=True, right_index=True)
    df_pivoted.rename(columns={'name': 'Country Name'}, inplace=True)
    
    return df_pivoted


def preprocess_data(df, selected_features=None, k_neighbors=5):
    """
    Impute, log transform, and scale data based on selected features.
    """
    if selected_features is None:
        selected_features = [col for col in df.columns if col != 'Country Name']
        
    df_selected = df[selected_features].copy()
    
    # Imputation using KNN
    imputer = KNNImputer(n_neighbors=k_neighbors)
    imputed_values = imputer.fit_transform(df_selected)
    df_imputed = pd.DataFrame(imputed_values, index=df_selected.index, columns=df_selected.columns)
    
    # Feature Transformation & Scaling
    # Log-Transform skewed features
    for col in df_imputed.columns:
        if col in SKEWED_FEATURES:
            # We want to log-transform, but only positive values. 
            # In our data, GDP and Pop Density are practically positive, but let's be safe:
            df_imputed[col] = np.log1p(np.maximum(df_imputed[col], 0))
            
    # Standardization
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df_imputed)
    df_scaled = pd.DataFrame(scaled_values, index=df_imputed.index, columns=df_imputed.columns)
    
    # Attach 'Country Name' back for the results dataframe
    if 'Country Name' in df.columns:
        df_imputed['Country Name'] = df['Country Name']
    
    return df_scaled, df_imputed, scaler
