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

@st.cache_data(show_spinner="Fetching historical indicator data from World Bank API...", persist="disk")
def load_data():
    """
    Fetches the historical indicator data (from 2010 to 2023) using wbgapi.
    Returns a longitudinal DataFrame containing metrics for each country per year.
    """
    indicators_list = list(INDICATORS.keys())
    
    # Fetch historical data from 2010 to 2023
    df = wb.data.DataFrame(indicators_list, time=range(2010, 2024))
    
    # Reset index (which is currently a MultiIndex of economy, series)
    df = df.reset_index()
    
    # Melt the dataframe from wide to long format
    df_long = pd.melt(df, id_vars=['economy', 'series'], var_name='year', value_name='value')
    
    # Clean up the 'year' column (e.g., from 'YR2010' to 2010)
    df_long['year'] = df_long['year'].str.replace('YR', '').astype(int)
    
    # Get economies metadata to filter real countries and attach full names
    economies = wb.economy.DataFrame()
    countries = economies[economies['aggregate'] == False]
    country_codes = countries.index.tolist()
    
    # Filter out regions, keeping only actual countries
    df_long = df_long[df_long['economy'].isin(country_codes)]
    
    # Pivot to make rows unique per (economy, year) and columns the indicators
    df_pivoted = df_long.pivot(index=['economy', 'year'], columns='series', values='value')
    df_pivoted.rename(columns=INDICATORS, inplace=True)
    
    # Flatten the index and merge Country Name
    df_pivoted = df_pivoted.reset_index()
    df_pivoted = df_pivoted.merge(countries[['name']], left_on='economy', right_index=True)
    df_pivoted.rename(columns={'name': 'Country Name'}, inplace=True)
    
    return df_pivoted


def preprocess_data(df, selected_features=None, k_neighbors=5):
    """
    Impute, log transform, and scale data on a per-year basis.
    Processes data year-by-year to prevent cross-year leakage and ensure scaling is relative to the specific year.
    """
    if selected_features is None:
        selected_features = [col for col in df.columns if col not in ['Country Name', 'economy', 'year']]
        
    scaled_dfs = []
    imputed_dfs = []
    
    # We group by year and process cross-sectional data strictly within that year
    for year, group in df.groupby('year'):
        df_year = group.set_index('Country Name')
        df_selected = df_year[selected_features].copy()
        
        # Imputation using KNN
        imputer = KNNImputer(n_neighbors=k_neighbors)
        # If a year's data is extremely sparse for some feature, KNN will still impute based on neighboring features
        imputed_values = imputer.fit_transform(df_selected)
        df_imputed_yr = pd.DataFrame(imputed_values, index=df_selected.index, columns=df_selected.columns)
        
        # Log-Transform skewed features
        for col in df_imputed_yr.columns:
            if col in SKEWED_FEATURES and col in selected_features:
                df_imputed_yr[col] = np.log1p(np.maximum(df_imputed_yr[col], 0))
                
        # Standardization
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(df_imputed_yr)
        df_scaled_yr = pd.DataFrame(scaled_values, index=df_imputed_yr.index, columns=df_imputed_yr.columns)
        
        # Re-attach metadata
        df_imputed_yr['year'] = year
        df_imputed_yr['economy'] = df_year['economy']
        
        df_scaled_yr['year'] = year
        df_scaled_yr['economy'] = df_year['economy']
        
        scaled_dfs.append(df_scaled_yr.reset_index())
        imputed_dfs.append(df_imputed_yr.reset_index())
        
    df_scaled_all = pd.concat(scaled_dfs, ignore_index=True)
    df_imputed_all = pd.concat(imputed_dfs, ignore_index=True)
    
    return df_scaled_all, df_imputed_all
