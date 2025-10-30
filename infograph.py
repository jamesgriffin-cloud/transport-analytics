import pandas as pd
import requests
import io
import json
import numpy as np
import datetime
import math
import sys
import time
import os

# --- Geocoding & Caching Setup ---
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

NOMINATIM_USER_AGENT = "PerrysTransportInfographicScript/1.0 (Contact: jamesgriffin@jigcar.com)"
GEOCODE_CACHE_FILE = "geocode_cache.json"

# Load existing cache
geocode_cache = {}
if os.path.exists(GEOCODE_CACHE_FILE):
    try:
        with open(GEOCODE_CACHE_FILE, 'r', encoding='utf-8') as f:
            geocode_cache = json.load(f)
        print(f"Loaded {len(geocode_cache)} items from geocode cache.")
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load geocode cache file '{GEOCODE_CACHE_FILE}'. Starting fresh. Error: {e}")
        geocode_cache = {}

# Initialize Nominatim geocoder
try:
    geolocator = Nominatim(user_agent=NOMINATIM_USER_AGENT)
    print(f"Nominatim geolocator initialized with user agent: {NOMINATIM_USER_AGENT}")
except Exception as e:
    print(f"CRITICAL ERROR initializing Nominatim geolocator: {e}")
    print("Geocoding will be skipped.")
    geolocator = None

# --- Configuration ---
GOOGLE_SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSJ8gfzvUcx9iIImufwA92PVZz5vxgd2i0ZvWQSUn_UNwkyHdYGIvgDotTl5lJO1SdK367HcREcz6wM/pub?output=csv"
OUTPUT_HTML_FILE = "perrys_transport_infographic.html"
DEFAULT_SLA_COLUMN = 'Days To Complete (Working)'
REGION_COLUMN = 'Business area'

# ========================================================================
# DATE FILTERING - Change these dates to analyze specific time periods
# ========================================================================
FILTER_START_DATE = None  # Example: '2024-01-01' for Jan 1st 2024
FILTER_END_DATE = None    # Example: '2024-12-31' for Dec 31st 2024
# ========================================================================

# Distance band configuration (in miles)
DISTANCE_BANDS = {
    'Under 25 Miles': (0, 25),
    '25 - 50': (25, 50),
    '50 - 100': (50, 100),
    '100 - 150': (100, 150),
    '150 Miles +': (150, float('inf'))
}

# --- Helper Functions ---
def clean_currency(value):
    if isinstance(value, (int, float)): return float(value)
    if pd.isna(value) or value in ['-', '#VALUE!', '']: return np.nan
    try:
        cleaned = str(value).replace('£', '').replace(',', '')
        return float(cleaned)
    except (ValueError, TypeError): return np.nan

def safe_mean(series):
    valid_series = series.dropna()
    if valid_series.empty: return np.nan
    return valid_series.mean()

def safe_sum(series):
    valid_series = series.dropna()
    if valid_series.empty: return 0.0
    return valid_series.sum()

def format_currency(value, default="N/A"):
    if pd.isna(value) or not isinstance(value, (int, float)): return default
    return f"£{value:,.2f}"

def format_number(value, decimals=1, default="N/A"):
    if pd.isna(value) or not isinstance(value, (int, float)): return default
    return f"{value:,.{decimals}f}"

def format_int(value, default="N/A"):
    if pd.isna(value): return default
    try: return f"{int(float(value)):,}"
    except (ValueError, TypeError): return default

def categorize_distance(distance):
    if pd.isna(distance):
        return 'Unknown'
    for band_name, (min_dist, max_dist) in DISTANCE_BANDS.items():
        if min_dist <= distance < max_dist:
            return band_name
    return 'Unknown'

def detect_outliers_iqr(series, multiplier=1.5):
    if series.empty or series.isna().all():
        return pd.Series([False] * len(series), index=series.index)
    
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return (series < lower_bound) | (series > upper_bound)

def get_coordinates(postcode, country_bias='GB'):
    if not geolocator or not postcode or pd.isna(postcode):
        return None

    postcode_norm = str(postcode).strip().upper()

    if postcode_norm in geocode_cache:
        return geocode_cache[postcode_norm]

    print(f"  Geocoding (API call): {postcode_norm}")
    try:
        location = geolocator.geocode(postcode_norm, country_codes=country_bias, timeout=10)
        time.sleep(1.1)
        
        if location:
            coords = (location.latitude, location.longitude)
            geocode_cache[postcode_norm] = coords
            return coords
        else:
            geocode_cache[postcode_norm] = None
            print(f"    -> Not found: {postcode_norm}")
            return None
    except GeocoderTimedOut:
        print(f"    -> Timed out: {postcode_norm}. Will retry next run.")
        return None
    except GeocoderServiceError as e:
        print(f"    -> Service error for {postcode_norm}: {e}. Will retry next run.")
        return None
    except Exception as e:
        print(f"    -> Unexpected error: {postcode_norm}: {e}")
        geocode_cache[postcode_norm] = None
        return None

def fetch_data(url):
    print(f"Fetching data from URL...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        csv_data = response.content.decode('utf-8')
        date_cols = ['Requested Date', 'Ready To Collect From', 'Delivery By',
                     'Estimated Delivery Date', 'Completed On']
        try:
            df = pd.read_csv(io.StringIO(csv_data), parse_dates=date_cols, dayfirst=True, low_memory=False)
        except (ValueError, KeyError) as date_err:
            print(f"Warning: Could not parse all date columns ({date_err}). Reading without date parsing.")
            df = pd.read_csv(io.StringIO(csv_data), low_memory=False)
            for col in date_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)

        print(f"Successfully fetched and parsed {len(df)} rows.")
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    except Exception as e:
        print(f"Error processing fetched data: {e}")
        return None

def preprocess_and_geocode(df):
    if df is None: return None
    print("Preprocessing data...")

    column_mapping = {
        'Final Cost': 'Price',
        'Miles': 'Distance',
        'Estimated Cost': 'Estimated_Cost_Raw'
    }
    
    for actual_col, expected_col in column_mapping.items():
        if actual_col in df.columns:
            df[expected_col] = df[actual_col]
            print(f"  Mapped '{actual_col}' → '{expected_col}'")

    df['Price_Num'] = df['Price'].apply(clean_currency)
    df['Estimated_Cost_Num'] = df.get('Estimated_Cost_Raw', pd.Series([np.nan]*len(df))).apply(clean_currency)
    df['Distance_Num'] = pd.to_numeric(df['Distance'], errors='coerce')
    
    df['CPM_Num'] = df.apply(
        lambda row: row['Price_Num'] / row['Distance_Num']
        if pd.notna(row['Price_Num']) and pd.notna(row['Distance_Num']) and row['Distance_Num'] > 0
        else np.nan,
        axis=1
    )
    print(f"  Calculated CPM for {df['CPM_Num'].notna().sum()} records")
    
    df['Cost_Variance'] = df.apply(
        lambda row: row['Price_Num'] - row['Estimated_Cost_Num']
        if pd.notna(row['Price_Num']) and pd.notna(row['Estimated_Cost_Num'])
        else np.nan,
        axis=1
    )
    df['Cost_Variance_Pct'] = df.apply(
        lambda row: ((row['Price_Num'] - row['Estimated_Cost_Num']) / row['Estimated_Cost_Num'] * 100)
        if pd.notna(row['Price_Num']) and pd.notna(row['Estimated_Cost_Num']) and row['Estimated_Cost_Num'] > 0
        else np.nan,
        axis=1
    )
    
    if DEFAULT_SLA_COLUMN in df.columns:
        df[DEFAULT_SLA_COLUMN] = pd.to_numeric(df[DEFAULT_SLA_COLUMN], errors='coerce')
    if 'Days To Complete (Total)' in df.columns:
        df['Days To Complete (Total)'] = pd.to_numeric(df['Days To Complete (Total)'], errors='coerce')

    df['Distance_Band'] = df['Distance_Num'].apply(categorize_distance)

    if geolocator:
        print("\n" + "="*70)
        print("GEOCODING POSTCODES")
        print("="*70)
        
        unique_postcodes = df['Delivery Postcode'].dropna().unique()
        total_postcodes = len(unique_postcodes)
        
        cached_count = sum(1 for pc in unique_postcodes if str(pc).strip().upper() in geocode_cache)
        uncached_count = total_postcodes - cached_count
        
        print(f"Found {total_postcodes} unique postcodes:")
        print(f"  - Already cached: {cached_count}")
        print(f"  - Need to geocode: {uncached_count}")
        
        if uncached_count > 0:
            estimated_time_sec = uncached_count * 1.1
            estimated_minutes = estimated_time_sec / 60
            print(f"  - Estimated time: ~{estimated_minutes:.1f} minutes")
        
        print("-"*70)

        postcode_coords = {}
        processed_count = 0
        api_call_count = 0
        start_time = time.time()
        
        for pc in unique_postcodes:
            coords = get_coordinates(pc)
            postcode_coords[str(pc).strip().upper()] = coords
            processed_count += 1
            
            pc_norm = str(pc).strip().upper()
            if pc_norm not in geocode_cache or geocode_cache[pc_norm] != coords:
                api_call_count += 1
            
            if processed_count % 10 == 0 or processed_count == total_postcodes:
                elapsed = time.time() - start_time
                progress_pct = (processed_count / total_postcodes) * 100
                print(f"  Progress: {processed_count}/{total_postcodes} ({progress_pct:.1f}%) - "
                      f"Elapsed: {elapsed/60:.1f}min - API calls: {api_call_count}")

        print("-"*70)
        print(f"Geocoding complete! Total time: {(time.time()-start_time)/60:.1f} minutes")
        print("="*70 + "\n")

        df['Coordinates'] = df['Delivery Postcode'].astype(str).str.strip().str.upper().map(postcode_coords)
        df['latitude'] = df['Coordinates'].apply(lambda x: x[0] if x else np.nan)
        df['longitude'] = df['Coordinates'].apply(lambda x: x[1] if x else np.nan)
        df = df.drop(columns=['Coordinates'])
        
        successful_geocodes = df['latitude'].notna().sum()
        print(f"Successfully geocoded {successful_geocodes}/{len(df)} rows ({(successful_geocodes/len(df)*100):.1f}%)")
    else:
        print("Skipping geocoding as geolocator failed to initialize.")
        df['latitude'] = np.nan
        df['longitude'] = np.nan

    print("Preprocessing finished.\n")
    return df

def calculate_analytics(df):
    if df is None: return {}
    print("Calculating analytics...")
    
    df_filtered = df.copy()
    date_filter_applied = False
    
    if FILTER_START_DATE or FILTER_END_DATE:
        if 'Completed On' in df_filtered.columns and pd.api.types.is_datetime64_any_dtype(df_filtered['Completed On']):
            original_count = len(df_filtered)
            
            if FILTER_START_DATE:
                start_date = pd.to_datetime(FILTER_START_DATE)
                df_filtered = df_filtered[df_filtered['Completed On'] >= start_date]
                date_filter_applied = True
                
            if FILTER_END_DATE:
                end_date = pd.to_datetime(FILTER_END_DATE)
                df_filtered = df_filtered[df_filtered['Completed On'] <= end_date]
                date_filter_applied = True
            
            if date_filter_applied:
                filtered_count = len(df_filtered)
                print(f"✓ Date filter applied: {original_count} rows → {filtered_count} rows")
    
    df_filtered_cost = df_filtered.dropna(subset=['Price_Num'])

    total_moves = len(df_filtered)
    total_cost = safe_sum(df_filtered_cost['Price_Num'])
    total_miles = safe_sum(df_filtered['Distance_Num'])
    avg_distance = safe_mean(df_filtered['Distance_Num'])
    overall_cpm = total_cost / total_miles if total_miles > 0 else np.nan
    avg_cost_per_car = total_cost / total_moves if total_moves > 0 else np.nan
    
    aborted_rows = df_filtered[df_filtered['Status'].astype(str).str.strip().str.lower() == 'aborted']
    aborts_cost = safe_sum(aborted_rows['Price_Num'])

    sla_col_to_use = None
    if DEFAULT_SLA_COLUMN in df_filtered.columns and df_filtered[DEFAULT_SLA_COLUMN].notna().any():
        sla_col_to_use = DEFAULT_SLA_COLUMN
    elif 'Days To Complete (Total)' in df_filtered.columns and df_filtered['Days To Complete (Total)'].notna().any():
        sla_col_to_use = 'Days To Complete (Total)'
    
    avg_sla = safe_mean(df_filtered[sla_col_to_use]) if sla_col_to_use else np.nan

    min_date, max_date, date_range_str = pd.NaT, pd.NaT, "N/A"
    if 'Completed On' in df_filtered.columns and pd.api.types.is_datetime64_any_dtype(df_filtered['Completed On']):
        valid_dates = df_filtered['Completed On'].dropna()
        if not valid_dates.empty:
            min_date, max_date = valid_dates.min(), valid_dates.max()
        if pd.notna(min_date) and pd.notna(max_date):
            try:
                date_range_str = f"{min_date.strftime('%d/%m/%Y')} - {max_date.strftime('%d/%m/%Y')}"
            except Exception:
                date_range_str = "Error Formatting Dates"

    summary_stats = {
        "total_moves": total_moves,
        "total_cost": total_cost,
        "total_miles": total_miles,
        "avg_distance": avg_distance,
        "overall_cpm": overall_cpm,
        "avg_cost_per_car": avg_cost_per_car,
        "aborts_cost": aborts_cost,
        "avg_sla": avg_sla,
        "data_date_range": date_range_str,
        "date_filter_applied": date_filter_applied
    }

    top_cpm = df_filtered_cost.sort_values('CPM_Num', ascending=False, na_position='last').head(5)
    top_price = df_filtered_cost.sort_values('Price_Num', ascending=False, na_position='last').head(5)
    cost_highlights = {
        "top_cpm": top_cpm[['Delivery Postcode', 'Distance_Num', 'Price_Num', 'CPM_Num', 'Driver / Provider']].to_dict(orient='records'),
        "top_price": top_price[['Delivery Postcode', 'Distance_Num', 'Price_Num', 'CPM_Num', 'Driver / Provider']].to_dict(orient='records')
    }

    top_10_cost_columns = ['VRM', 'Delivery Postcode', 'Distance_Num', 'Price_Num', 'CPM_Num',
                           'Driver / Provider', 'Movement Type', 'Completed On']
    available_columns = [col for col in top_10_cost_columns if col in df_filtered_cost.columns]
    
    top_10_by_cost = df_filtered_cost.sort_values('Price_Num', ascending=False, na_position='last').head(10)
    top_10_cost_data = top_10_by_cost[available_columns].to_dict(orient='records')
    
    top_10_by_cpm = df_filtered_cost.sort_values('CPM_Num', ascending=False, na_position='last').head(10)
    top_10_cpm_data = top_10_by_cpm[available_columns].to_dict(orient='records')

    # On-Time Performance by Day of Week
    ontime_by_day_analysis = {}
    if 'On Time' in df_filtered.columns and 'Completed On' in df_filtered.columns:
        df_ontime = df_filtered[df_filtered['Completed On'].notna()].copy()
        if not df_ontime.empty:
            df_ontime['is_ontime'] = df_ontime['On Time'].astype(str).str.strip().str.lower() == 'on time'
            df_ontime['Year_Month'] = df_ontime['Completed On'].dt.to_period('M')
            df_ontime['Day_of_Week'] = df_ontime['Completed On'].dt.day_name()
            
            ontime_pivot = df_ontime.groupby(['Year_Month', 'Day_of_Week']).agg(
                total=('is_ontime', 'count'),
                ontime=('is_ontime', 'sum')
            ).reset_index()
            ontime_pivot['percentage'] = (ontime_pivot['ontime'] / ontime_pivot['total'] * 100).round(1)
            ontime_pivot['period'] = ontime_pivot['Year_Month'].astype(str)
            
            overall_by_month = df_ontime.groupby('Year_Month').agg(
                total=('is_ontime', 'count'),
                ontime=('is_ontime', 'sum')
            ).reset_index()
            overall_by_month['percentage'] = (overall_by_month['ontime'] / overall_by_month['total'] * 100).round(1)
            overall_by_month['period'] = overall_by_month['Year_Month'].astype(str)
            overall_by_month['Day_of_Week'] = 'Overall'
            
            combined_ontime = pd.concat([
                ontime_pivot[['period', 'Day_of_Week', 'percentage', 'total']],
                overall_by_month[['period', 'Day_of_Week', 'percentage', 'total']]
            ])
            
            ontime_by_day_analysis = {
                'data': combined_ontime.to_dict(orient='records'),
                'months': sorted(combined_ontime['period'].unique().tolist()),
                'days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'Overall']
            }
            print(f"✓ On-time by day analysis complete: {len(ontime_by_day_analysis['months'])} months analyzed")

    time_trends = {}
    if 'Completed On' in df_filtered.columns and pd.api.types.is_datetime64_any_dtype(df_filtered['Completed On']):
        df_with_dates = df_filtered[df_filtered['Completed On'].notna()].copy()
        if not df_with_dates.empty:
            df_with_dates['Year_Month'] = df_with_dates['Completed On'].dt.to_period('M')
            df_with_dates['Quarter'] = df_with_dates['Completed On'].dt.to_period('Q')
            
            monthly = df_with_dates.groupby('Year_Month').agg(
                volume=('VRM', 'count'),
                total_cost=('Price_Num', lambda x: safe_sum(x)),
                avg_cost=('Price_Num', safe_mean),
                avg_distance=('Distance_Num', safe_mean)
            ).reset_index()
            monthly['period'] = monthly['Year_Month'].astype(str)
            
            quarterly = df_with_dates.groupby('Quarter').agg(
                volume=('VRM', 'count'),
                total_cost=('Price_Num', lambda x: safe_sum(x)),
                avg_cost=('Price_Num', safe_mean)
            ).reset_index()
            quarterly['period'] = quarterly['Quarter'].astype(str)
            
            time_trends = {
                'monthly': monthly[['period', 'volume', 'total_cost', 'avg_cost', 'avg_distance']].to_dict(orient='records'),
                'quarterly': quarterly[['period', 'volume', 'total_cost', 'avg_cost']].to_dict(orient='records')
            }

    volume_aborts_monthly = []
    if 'Completed On' in df_filtered.columns and pd.api.types.is_datetime64_any_dtype(df_filtered['Completed On']):
        df_vol_aborts = df_filtered[df_filtered['Completed On'].notna()].copy()
        if not df_vol_aborts.empty:
            df_vol_aborts['Year_Month'] = df_vol_aborts['Completed On'].dt.to_period('M')
            df_vol_aborts['is_aborted'] = df_vol_aborts['Status'].astype(str).str.strip().str.lower() == 'aborted'
            
            monthly_vol_aborts = df_vol_aborts.groupby('Year_Month').agg(
                total_movements=('VRM', 'count'),
                aborts=('is_aborted', 'sum')
            ).reset_index()
            
            monthly_vol_aborts['abort_pct'] = (monthly_vol_aborts['aborts'] / monthly_vol_aborts['total_movements'] * 100)
            monthly_vol_aborts['period'] = monthly_vol_aborts['Year_Month'].astype(str)
            monthly_vol_aborts['movements'] = monthly_vol_aborts['total_movements'] - monthly_vol_aborts['aborts']
            
            volume_aborts_monthly = monthly_vol_aborts[['period', 'movements', 'aborts', 'abort_pct', 'total_movements']].to_dict(orient='records')

    provider_performance = []
    if 'Driver / Provider' in df_filtered.columns:
        if 'On Time' in df_filtered.columns:
            df_filtered['is_ontime_from_field'] = df_filtered['On Time'].astype(str).str.strip().str.lower() == 'on time'
            ontime_field = 'is_ontime_from_field'
        else:
            df_ontime = df_filtered.copy()
            if 'Completed On' in df_ontime.columns and 'Delivery By' in df_ontime.columns:
                df_ontime['is_ontime'] = (df_ontime['Completed On'] <= df_ontime['Delivery By'])
                ontime_field = 'is_ontime'
            else:
                df_ontime['is_ontime'] = np.nan
                ontime_field = 'is_ontime'
        
        provider_groups = df_filtered.groupby('Driver / Provider').agg(
            total_moves=('VRM', 'count'),
            avg_cost=('Price_Num', safe_mean),
            avg_cpm=('CPM_Num', safe_mean),
            total_cost=('Price_Num', lambda x: safe_sum(x))
        ).reset_index()
        
        ontime_by_provider = df_filtered.groupby('Driver / Provider')[ontime_field].apply(
            lambda x: (x.sum() / x.count() * 100) if x.notna().any() else np.nan
        )
        provider_groups['ontime_pct'] = provider_groups['Driver / Provider'].map(ontime_by_provider)
        
        if not pd.isna(overall_cpm) and overall_cpm > 0:
            provider_groups['efficiency_rating'] = (overall_cpm / provider_groups['avg_cpm']) * 100
        else:
            provider_groups['efficiency_rating'] = np.nan
        
        provider_groups = provider_groups.sort_values('total_moves', ascending=False).head(10)
        provider_performance = provider_groups.to_dict(orient='records')

    distance_analysis = []
    if 'Distance_Band' in df_filtered.columns:
        band_groups = df_filtered.groupby('Distance_Band').agg(
            volume=('VRM', 'count'),
            total_cost=('Price_Num', lambda x: safe_sum(x)),
            avg_cost=('Price_Num', safe_mean),
            avg_cpm=('CPM_Num', safe_mean),
            avg_distance=('Distance_Num', safe_mean)
        ).reset_index()
        
        band_order = list(DISTANCE_BANDS.keys()) + ['Unknown']
        band_groups['Distance_Band'] = pd.Categorical(band_groups['Distance_Band'], categories=band_order, ordered=True)
        band_groups = band_groups.sort_values('Distance_Band')
        distance_analysis = band_groups.to_dict(orient='records')

    outliers_data = []
    if not df_filtered_cost.empty and len(df_filtered_cost) >= 10:
        df_outliers = df_filtered_cost.copy()
        df_outliers['is_price_outlier'] = detect_outliers_iqr(df_outliers['Price_Num'])
        df_outliers['is_cpm_outlier'] = detect_outliers_iqr(df_outliers['CPM_Num'])
        
        outliers = df_outliers[df_outliers['is_price_outlier'] | df_outliers['is_cpm_outlier']].copy()
        outliers['outlier_type'] = outliers.apply(
            lambda row: 'Both' if row['is_price_outlier'] and row['is_cpm_outlier']
            else ('Price' if row['is_price_outlier'] else 'CPM'), axis=1
        )
        
        outliers_sorted = outliers.sort_values('Price_Num', ascending=False).head(20)
        outliers_data = outliers_sorted[['Delivery Postcode', 'Distance_Num', 'Price_Num', 'CPM_Num',
                                          'Driver / Provider', 'outlier_type']].to_dict(orient='records')

    seasonal_patterns = {}
    if 'Completed On' in df_filtered.columns and pd.api.types.is_datetime64_any_dtype(df_filtered['Completed On']):
        df_seasonal = df_filtered[df_filtered['Completed On'].notna()].copy()
        if not df_seasonal.empty:
            df_seasonal['Month'] = df_seasonal['Completed On'].dt.month
            df_seasonal['Month_Name'] = df_seasonal['Completed On'].dt.strftime('%B')
            
            seasonal = df_seasonal.groupby(['Month', 'Month_Name']).agg(
                avg_volume=('VRM', 'count'),
                avg_cost=('Price_Num', safe_mean),
                total_cost=('Price_Num', lambda x: safe_sum(x))
            ).reset_index()
            
            years_span = (df_seasonal['Completed On'].max() - df_seasonal['Completed On'].min()).days / 365.25
            if years_span > 0:
                seasonal['avg_volume'] = seasonal['avg_volume'] / max(years_span, 1)
            
            seasonal = seasonal.sort_values('Month')
            seasonal_patterns = seasonal[['Month_Name', 'avg_volume', 'avg_cost', 'total_cost']].to_dict(orient='records')

    sankey_data = {}
    try:
        if 'Business area' in df_filtered.columns and 'Transport Type' in df_filtered.columns and 'Driver / Provider' in df_filtered.columns:
            df_sankey = df_filtered[df_filtered['Price_Num'].notna()].copy()
            
            if not df_sankey.empty and len(df_sankey) > 0:
                df_sankey['Business area'] = df_sankey['Business area'].fillna('Unknown').astype(str)
                df_sankey['Transport Type'] = df_sankey['Transport Type'].fillna('Unknown').astype(str)
                df_sankey['Driver / Provider'] = df_sankey['Driver / Provider'].fillna('Unknown').astype(str)
                
                nodes = []
                links = []
                node_dict = {}
                
                def get_node_index(name, node_dict, nodes):
                    if name not in node_dict:
                        node_dict[name] = len(nodes)
                        nodes.append(name)
                    return node_dict[name]
                
                business_to_transport = df_sankey.groupby(['Business area', 'Transport Type'])['Price_Num'].sum().reset_index()
                for _, row in business_to_transport.iterrows():
                    if row['Price_Num'] > 0:
                        source_idx = get_node_index(row['Business area'], node_dict, nodes)
                        target_idx = get_node_index(row['Transport Type'], node_dict, nodes)
                        links.append({
                            'source': source_idx,
                            'target': target_idx,
                            'value': float(row['Price_Num'])
                        })
                
                transport_to_provider = df_sankey.groupby(['Transport Type', 'Driver / Provider'])['Price_Num'].sum().reset_index()
                top_providers = df_sankey.groupby('Driver / Provider')['Price_Num'].sum().nlargest(8).index.tolist()
                transport_to_provider_top = transport_to_provider[transport_to_provider['Driver / Provider'].isin(top_providers)]
                
                for _, row in transport_to_provider_top.iterrows():
                    if row['Price_Num'] > 0:
                        source_idx = get_node_index(row['Transport Type'], node_dict, nodes)
                        target_idx = get_node_index(row['Driver / Provider'], node_dict, nodes)
                        links.append({
                            'source': source_idx,
                            'target': target_idx,
                            'value': float(row['Price_Num'])
                        })
                
                for transport_type in df_sankey['Transport Type'].unique():
                    other_cost = df_sankey[
                        (df_sankey['Transport Type'] == transport_type) &
                        (~df_sankey['Driver / Provider'].isin(top_providers))
                    ]['Price_Num'].sum()
                    
                    if other_cost > 0:
                        source_idx = get_node_index(transport_type, node_dict, nodes)
                        target_idx = get_node_index('Other Providers', node_dict, nodes)
                        links.append({
                            'source': source_idx,
                            'target': target_idx,
                            'value': float(other_cost)
                        })
                
                if len(nodes) > 0 and len(links) > 0:
                    sankey_data = {
                        'nodes': nodes,
                        'links': links,
                        'total_cost': float(safe_sum(df_sankey['Price_Num']))
                    }
                    print(f"✓ Sankey data prepared: {len(nodes)} nodes, {len(links)} links")
    except Exception as e:
        print(f"Warning: Could not generate Sankey data: {e}")
        sankey_data = {}

    budget_analysis = {}
    df_with_variance = df_filtered[df_filtered['Cost_Variance'].notna()].copy()
    if not df_with_variance.empty:
        total_estimated = safe_sum(df_with_variance['Estimated_Cost_Num'])
        total_actual = safe_sum(df_with_variance['Price_Num'])
        total_variance = total_actual - total_estimated
        
        over_budget_count = len(df_with_variance[df_with_variance['Cost_Variance'] > 0])
        under_budget_count = len(df_with_variance[df_with_variance['Cost_Variance'] < 0])
        on_budget_count = len(df_with_variance[df_with_variance['Cost_Variance'] == 0])
        
        avg_variance_pct = safe_mean(df_with_variance['Cost_Variance_Pct'])
        
        top_over_budget = df_with_variance[df_with_variance['Cost_Variance'] > 0].sort_values(
            'Cost_Variance', ascending=False
        ).head(10)[['VRM', 'Delivery Postcode', 'Estimated_Cost_Num', 'Price_Num',
                    'Cost_Variance', 'Cost_Variance_Pct', 'Driver / Provider']].to_dict(orient='records')
        
        budget_analysis = {
            'total_estimated': total_estimated,
            'total_actual': total_actual,
            'total_variance': total_variance,
            'over_budget_count': over_budget_count,
            'under_budget_count': under_budget_count,
            'on_budget_count': on_budget_count,
            'avg_variance_pct': avg_variance_pct,
            'records_analyzed': len(df_with_variance),
            'top_over_budget': top_over_budget
        }

    shift_analysis = {}
    if 'Transport Type' in df_filtered.columns and 'Driver / Provider' in df_filtered.columns:
        internal_drivers = df_filtered[
            df_filtered['Transport Type'].astype(str).str.strip().str.lower().str.contains('internal', na=False)
        ].copy()
        
        if not internal_drivers.empty and 'Completed On' in internal_drivers.columns:
            internal_drivers['Completion_Date'] = internal_drivers['Completed On'].dt.date
            
            shifts = internal_drivers.groupby(['Driver / Provider', 'Completion_Date']).agg(
                moves_in_shift=('VRM', 'count'),
                total_cost=('Price_Num', lambda x: safe_sum(x)),
                total_miles=('Distance_Num', lambda x: safe_sum(x))
            ).reset_index()
            
            # === DAY OF WEEK ANALYSIS (NEW) ===
            internal_drivers['Day_of_Week'] = internal_drivers['Completed On'].dt.day_name()
            day_of_week_analysis = internal_drivers.groupby('Day_of_Week').agg(
                total_shifts=('Completion_Date', lambda x: x.nunique()),
                total_cars_moved=('VRM', 'count')
            ).reset_index()
            
            day_of_week_analysis['avg_moves_per_shift'] = (
                day_of_week_analysis['total_cars_moved'] / day_of_week_analysis['total_shifts']
            )
            
            # Reorder days properly
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_of_week_analysis['Day_of_Week'] = pd.Categorical(
                day_of_week_analysis['Day_of_Week'],
                categories=day_order,
                ordered=True
            )
            day_of_week_analysis = day_of_week_analysis.sort_values('Day_of_Week')
            # === END OF DAY OF WEEK ANALYSIS ===
            
            driver_averages = shifts.groupby('Driver / Provider').agg(
                total_shifts=('moves_in_shift', 'count'),
                avg_moves_per_shift=('moves_in_shift', 'mean'),
                total_moves=('moves_in_shift', 'sum'),
                avg_cost_per_shift=('total_cost', 'mean'),
                avg_miles_per_shift=('total_miles', 'mean')
            ).reset_index()
            
            driver_averages = driver_averages.sort_values('avg_moves_per_shift', ascending=False).head(15)
            
            internal_drivers['Year_Month'] = internal_drivers['Completed On'].dt.to_period('M')
            monthly_shifts = internal_drivers.groupby(['Year_Month', 'Driver / Provider', 'Completion_Date']).size().reset_index(name='moves')
            monthly_avg = monthly_shifts.groupby('Year_Month')['moves'].mean().reset_index()
            monthly_avg.columns = ['period', 'avg_moves_per_shift']
            monthly_avg['period'] = monthly_avg['period'].astype(str)
            
            overall_avg_moves = shifts['moves_in_shift'].mean()
            total_shifts_analyzed = len(shifts)
            total_drivers = internal_drivers['Driver / Provider'].nunique()
            
            shift_analysis = {
                'day_of_week_analysis': day_of_week_analysis.to_dict(orient='records'),  # NEW LINE
                'driver_performance': driver_averages.to_dict(orient='records'),
                'monthly_trend': monthly_avg.to_dict(orient='records'),
                'overall_avg_moves': overall_avg_moves,
                'total_shifts_analyzed': total_shifts_analyzed,
                'total_drivers': total_drivers,
                'shifts_data': shifts[['Driver / Provider', 'Completion_Date', 'moves_in_shift']].to_dict(orient='records')[:50]
            }
            print(f"✓ Shift analysis complete: {total_drivers} drivers, {total_shifts_analyzed} shifts analyzed")

    speed_analysis = {}
    if sla_col_to_use and 'Completed On' in df_filtered.columns and 'Driver / Provider' in df_filtered.columns:
        df_speed = df_filtered[df_filtered[sla_col_to_use].notna()].copy()
        
        if not df_speed.empty:
            df_speed['Year_Month'] = df_speed['Completed On'].dt.to_period('M')
            
            monthly_sla = df_speed.groupby('Year_Month').agg(
                avg_days=(sla_col_to_use, safe_mean),
                volume=('VRM', 'count')
            ).reset_index()
            monthly_sla['period'] = monthly_sla['Year_Month'].astype(str)
            
            top_providers = df_speed['Driver / Provider'].value_counts().head(5).index.tolist()
            df_speed_top = df_speed[df_speed['Driver / Provider'].isin(top_providers)].copy()
            
            provider_monthly = df_speed_top.groupby(['Year_Month', 'Driver / Provider']).agg(
                avg_days=(sla_col_to_use, safe_mean)
            ).reset_index()
            provider_monthly['period'] = provider_monthly['Year_Month'].astype(str)
            
            provider_averages = df_speed.groupby('Driver / Provider').agg(
                avg_days=(sla_col_to_use, safe_mean),
                volume=('VRM', 'count')
            ).reset_index()
            provider_averages = provider_averages.sort_values('volume', ascending=False).head(10)
            
            overall_avg_sla = safe_mean(df_speed[sla_col_to_use])
            
            speed_analysis = {
                'monthly_overall': monthly_sla[['period', 'avg_days', 'volume']].to_dict(orient='records'),
                'provider_monthly': provider_monthly.to_dict(orient='records'),
                'provider_averages': provider_averages.to_dict(orient='records'),
                'overall_avg_sla': overall_avg_sla,
                'top_providers': top_providers,
                'sla_column_name': sla_col_to_use
            }
            print(f"✓ Speed analysis complete: {len(monthly_sla)} months analyzed")

    transport_type_analysis = []
    if 'Transport Type' in df_filtered.columns:
        transport_groups = df_filtered.groupby('Transport Type').agg(
            volume=('VRM', 'count'),
            total_cost=('Price_Num', lambda x: safe_sum(x)),
            avg_cost=('Price_Num', safe_mean),
            avg_cpm=('CPM_Num', safe_mean),
            avg_distance=('Distance_Num', safe_mean)
        ).reset_index()
        
        if 'On Time' in df_filtered.columns:
            df_filtered['is_ontime_calc'] = df_filtered['On Time'].astype(str).str.strip().str.lower() == 'on time'
            ontime_by_type = df_filtered.groupby('Transport Type')['is_ontime_calc'].apply(
                lambda x: (x.sum() / x.count() * 100) if x.notna().any() else np.nan
            )
            transport_groups['ontime_pct'] = transport_groups['Transport Type'].map(ontime_by_type)
        else:
            transport_groups['ontime_pct'] = np.nan
        
        transport_groups = transport_groups.sort_values('volume', ascending=False)
        transport_type_analysis = transport_groups.to_dict(orient='records')

    regional_data = []
    if REGION_COLUMN in df_filtered.columns:
        df_filtered[REGION_COLUMN] = df_filtered[REGION_COLUMN].fillna('Unknown').astype(str)
        agg_dict = {'moves': ('VRM', 'count'), 'avg_cost': ('Price_Num', safe_mean)}
        if sla_col_to_use:
            agg_dict['avg_sla'] = (sla_col_to_use, safe_mean)
        regional_groups = df_filtered.groupby(REGION_COLUMN).agg(**agg_dict).reset_index()
        if not sla_col_to_use:
            regional_groups['avg_sla'] = np.nan
        regional_groups = regional_groups.sort_values('moves', ascending=False)
        regional_data = regional_groups.to_dict(orient='records')

    status_counts = df_filtered['Status'].astype(str).fillna('Unknown').value_counts().to_dict()
    movement_type_counts = df_filtered['Movement Type'].astype(str).fillna('Unknown').value_counts().head(6).to_dict()
    provider_counts = df_filtered['Driver / Provider'].astype(str).fillna('Unknown').value_counts().head(5).to_dict()
    method_counts = df_filtered['Transport Method'].astype(str).fillna('Unknown').value_counts().to_dict()
    stock_counts = df_filtered['Stock'].astype(str).fillna('Unknown').value_counts().to_dict()
    heatmap_points = df_filtered[['latitude', 'longitude', 'Price_Num']].dropna().values.tolist()

    def make_json_safe(data):
        if isinstance(data, list):
            return [make_json_safe(item) for item in data]
        elif isinstance(data, dict):
            return {k: make_json_safe(v) for k, v in data.items()}
        elif isinstance(data, (float, np.floating)):
            return None if not np.isfinite(data) else data
        elif isinstance(data, (int, np.integer, np.int64)):
            return int(data)
        elif isinstance(data, pd.Timestamp):
            return data.isoformat()
        elif isinstance(data, (pd.Period, pd.Interval)):
            return str(data)
        elif hasattr(data, 'isoformat'):
            return data.isoformat()
        else:
            return data

    chart_map_data = make_json_safe({
        "status_counts": status_counts,
        "movement_type_counts": movement_type_counts,
        "provider_counts": provider_counts,
        "method_counts": method_counts,
        "stock_counts": stock_counts,
        "regional_data": regional_data,
        "heatmap_points": heatmap_points,
        "region_column_name": REGION_COLUMN,
        "time_trends": time_trends,
        "volume_aborts_monthly": volume_aborts_monthly,
        "provider_performance": provider_performance,
        "distance_analysis": distance_analysis,
        "outliers": outliers_data,
        "seasonal_patterns": seasonal_patterns,
        "top_10_by_cost": top_10_cost_data,
        "top_10_by_cpm": top_10_cpm_data,
        "budget_analysis": budget_analysis,
        "transport_type_analysis": transport_type_analysis,
        "sankey_data": sankey_data,
        "shift_analysis": shift_analysis,
        "speed_analysis": speed_analysis,
        "ontime_by_day": ontime_by_day_analysis
    })

    analytics = {
        "summary_stats": summary_stats,
        "cost_highlights": cost_highlights,
        "chart_map_data": chart_map_data
    }
    
    print("Analytics calculation complete.\n")
    return analytics

def generate_html_infographic(analytics_data, output_file):
    """Generates the comprehensive HTML infographic file with all analytics."""
    if not analytics_data:
        print("No analytics data to generate HTML.")
        return

    print(f"Generating HTML file: {output_file}")
    summary = analytics_data['summary_stats']
    highlights = analytics_data['cost_highlights']
    chart_map_js_data = json.dumps(analytics_data['chart_map_data'], allow_nan=False, default=lambda x: None if pd.isna(x) else x)

    # Build cost highlights HTML
    top_cpm_html = ""
    for item in highlights.get('top_cpm', []):
        cpm_val = format_currency(item.get('CPM_Num'), default='N/A')
        dist_val = format_number(item.get('Distance_Num'), 1, 'N/A')
        provider = item.get('Driver / Provider', 'N/A')
        top_cpm_html += f"<p><strong>{cpm_val}/mile</strong> ({dist_val} miles) via <span class='provider'>{provider if provider else 'N/A'}</span></p>"
    if not top_cpm_html:
        top_cpm_html = "<p>No CPM data available.</p>"

    top_price_html = ""
    for item in highlights.get('top_price', []):
        price_val = format_currency(item.get('Price_Num'), default='N/A')
        dist_val = format_number(item.get('Distance_Num'), 1, 'N/A')
        provider = item.get('Driver / Provider', 'N/A')
        top_price_html += f"<p><strong>{price_val}</strong> ({dist_val} miles) via <span class='provider'>{provider if provider else 'N/A'}</span></p>"
    if not top_price_html:
        top_price_html = "<p>No Price data available.</p>"

    # Date filter notice
    date_filter_notice = ""
    if summary.get('date_filter_applied', False):
        start_display = FILTER_START_DATE if FILTER_START_DATE else "No start limit"
        end_display = FILTER_END_DATE if FILTER_END_DATE else "No end limit"
        date_filter_notice = f"""
        <div style="background-color: #fef3c7; border: 1px solid #f59e0b; border-radius: 8px; padding: 16px; margin-bottom: 24px;">
            <p style="margin: 0; color: #92400e; font-weight: 600;">
                📅 Date Filter Active: {start_display} to {end_display}
            </p>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Perrys Transport Analytics - Comprehensive Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin=""/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>
    <script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        :root {{
            --jigcar-green: #34d399;
            --jigcar-green-dark: #10b981;
            --jigcar-green-light: #d1fae5;
            --jigcar-cyan: #06b6d4;
            --jigcar-black: #111827;
            --jigcar-gray-dark: #4b5563;
            --jigcar-gray: #9ca3af;
            --jigcar-gray-light: #f3f4f6;
            --jigcar-white: #ffffff;
            --jigcar-border: #e5e7eb;
        }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background-color: var(--jigcar-gray-light);
            color: var(--jigcar-black);
            margin: 0;
            padding: 20px;
            line-height: 1.6;
            -webkit-font-smoothing: antialiased;
        }}
        
        .infographic-container {{
            max-width: 1400px;
            margin: 20px auto;
            padding: 0;
        }}
        
        header h1 {{
            text-align: left;
            color: var(--jigcar-black);
            margin-bottom: 8px;
            font-weight: 700;
            font-size: 2.25em;
            letter-spacing: -0.025em;
        }}
        
        header .subtitle {{
            text-align: left;
            color: var(--jigcar-gray-dark);
            margin-bottom: 40px;
            font-size: 1em;
        }}
        
        .section-header {{
            font-size: 1.5em;
            color: var(--jigcar-black);
            margin: 50px 0 20px 0;
            font-weight: 600;
            letter-spacing: -0.025em;
        }}
        
        .grid-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 16px;
            margin-bottom: 24px;
        }}
        
        .card {{
            background-color: var(--jigcar-white);
            border-radius: 12px;
            padding: 24px;
            border: 1px solid var(--jigcar-border);
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
            transition: box-shadow 0.2s ease;
            overflow: hidden;
        }}
        
        .card:hover {{
            box-shadow: 0 4px 6px rgba(52, 211, 153, 0.1);
        }}
        
        .card h2 {{
            font-size: 1.125em;
            color: var(--jigcar-black);
            margin-top: 0;
            margin-bottom: 20px;
            font-weight: 600;
            letter-spacing: -0.01em;
        }}
        
        .chart-container {{
            position: relative;
            height: 250px;
            width: 100%;
            margin-bottom: 10px;
        }}
        
        .chart-container-large {{
            position: relative;
            height: 350px;
            width: 100%;
            margin-bottom: 10px;
        }}
        
        #mapHeatmap {{
            height: 400px;
            width: 100%;
            border-radius: 12px;
            border: 1px solid var(--jigcar-border);
            margin-top: 10px;
        }}
        
        #sankeyDiagram {{
            width: 100%;
            height: 600px;
            border-radius: 8px;
        }}
        
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
            gap: 16px;
            text-align: left;
        }}
        
        .kpi .value {{
            font-size: 2em;
            font-weight: 700;
            color: var(--jigcar-black);
            display: block;
            margin-bottom: 4px;
            letter-spacing: -0.02em;
        }}
        
        .kpi .label {{
            font-size: 0.875em;
            color: var(--jigcar-gray-dark);
            line-height: 1.4;
        }}
        
        .list-section ul {{
            list-style: none;
            padding: 0;
            margin: 0;
        }}
        
        .list-section li, .text-content p {{
            background-color: var(--jigcar-gray-light);
            border-left: 3px solid var(--jigcar-green);
            border-radius: 6px;
            padding: 12px 16px;
            margin-bottom: 8px;
            font-size: 0.9em;
        }}
        
        .text-content p {{
            background-color: transparent;
            border: none;
            padding: 4px 0;
            margin-bottom: 6px;
        }}
        
        .text-content p.highlight {{
            font-weight: 600;
            color: var(--jigcar-black);
            margin-top: 16px;
            margin-bottom: 8px;
        }}
        
        .text-content .provider {{
            color: var(--jigcar-green-dark);
            font-weight: 500;
        }}
        
        .table-container {{
            overflow-x: auto;
            max-height: 500px;
            overflow-y: auto;
            border-radius: 12px;
            border: 1px solid var(--jigcar-border);
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.875em;
        }}
        
        th {{
            background-color: var(--jigcar-gray-light);
            color: var(--jigcar-black);
            padding: 12px 16px;
            text-align: left;
            position: sticky;
            top: 0;
            z-index: 10;
            font-weight: 600;
            font-size: 0.8em;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            border-bottom: 1px solid var(--jigcar-border);
        }}
        
        td {{
            padding: 12px 16px;
            border-bottom: 1px solid var(--jigcar-border);
            color: var(--jigcar-gray-dark);
        }}
        
        tr:hover {{
            background-color: #fafafa;
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.75em;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        .badge-high {{
            background-color: #fee2e2;
            color: #991b1b;
        }}
        
        .badge-medium {{
            background-color: #fef3c7;
            color: #92400e;
        }}
        
        .badge-low {{
            background-color: var(--jigcar-green-light);
            color: var(--jigcar-green-dark);
        }}
        
        .ontime-heatmap {{
            width: 100%;
            overflow-x: auto;
            margin-top: 10px;
        }}
        
        .ontime-heatmap table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }}
        
        .ontime-heatmap th {{
            background-color: var(--jigcar-black);
            color: white;
            padding: 12px 8px;
            text-align: center;
            font-weight: 600;
            font-size: 0.85em;
        }}
        
        .ontime-heatmap td {{
            padding: 12px 8px;
            text-align: center;
            font-weight: 600;
            border: 1px solid #ddd;
        }}
        
        .ontime-heatmap td.month-label {{
            background-color: var(--jigcar-gray-light);
            font-weight: 600;
            text-align: left;
            padding-left: 16px;
        }}
        
        .ontime-cell-high {{
            background-color: #10b981;
            color: white;
        }}
        
        .ontime-cell-medium {{
            background-color: #fca5a5;
            color: var(--jigcar-black);
        }}
        
        .ontime-cell-low {{
            background-color: #dc2626;
            color: white;
        }}
        
        footer {{
            text-align: center;
            margin-top: 60px;
            font-size: 0.875em;
            color: var(--jigcar-gray);
            padding-top: 24px;
            border-top: 1px solid var(--jigcar-border);
        }}
        
        @media (max-width: 768px) {{
            .grid-container {{ grid-template-columns: 1fr; }}
            .kpi-grid {{ grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); }}
            header h1 {{ font-size: 1.875em; }}
        }}
    </style>
</head>
<body>
    <div class="infographic-container">
        <header>
            <h1>🚛 Transport Analytics Dashboard</h1>
            <p class="subtitle">Comprehensive Performance Analysis | Data Period: {summary.get('data_date_range', 'N/A')}</p>
        </header>
        
        {date_filter_notice}

        <h2 class="section-header">📊 Executive Summary</h2>
        <div class="summary-grid">
            <div class="card">
                <div class="kpi-grid">
                    <div class="kpi">
                        <span class="value">{format_int(summary.get('total_moves'), default='0')}</span>
                        <span class="label">Total Moves<br/>Completed</span>
                    </div>
                    <div class="kpi">
                        <span class="value">{format_currency(summary.get('total_cost'), default='£0')}</span>
                        <span class="label">Total<br/>Costs</span>
                    </div>
                    <div class="kpi">
                        <span class="value">{format_int(summary.get('total_miles'), default='0')}</span>
                        <span class="label">Total<br/>Miles</span>
                    </div>
                    <div class="kpi">
                        <span class="value">{format_number(summary.get('avg_distance'), 1)} mi</span>
                        <span class="label">Average<br/>Distance</span>
                    </div>
                    <div class="kpi">
                        <span class="value">{format_currency(summary.get('overall_cpm'))}</span>
                        <span class="label">Overall<br/>Cost/Mile</span>
                    </div>
                    <div class="kpi">
                        <span class="value">{format_currency(summary.get('avg_cost_per_car'))}</span>
                        <span class="label">Average<br/>Cost/Car</span>
                    </div>
                    <div class="kpi">
                        <span class="value">{format_currency(summary.get('aborts_cost'), default='£0')}</span>
                        <span class="label">Aborted<br/>Movements Cost</span>
                    </div>
                    <div class="kpi">
                        <span class="value">{format_number(summary.get('avg_sla'), 1)} days</span>
                        <span class="label">Average working<br/>days to complete</span>
                    </div>
                </div>
            </div>
        </div>

        <h2 class="section-header">💰 Cost Flow Breakdown</h2>
        <div class="card">
            <h2>Transport Cost Flow: Business Area → Transport Type → Provider</h2>
            <p style="font-size: 0.9em; color: #6b7280; margin-bottom: 16px;">
                Visual breakdown showing how transport costs flow through business areas, transport types, and providers
            </p>
            <div id="sankeyDiagram"></div>
        </div>

        <div class="summary-grid" style="margin-top: 24px;">
            <div class="card">
                <h2>🗺️ Delivery Cost Heatmap</h2>
                <p style="font-size: 0.9em; color: #6b7280; margin-bottom: 10px;">
                    Visual representation of delivery locations and costs. Intensity indicates price (darker/redder = higher cost).
                    <strong style="color: #059669;">Geocoded using Nominatim (OpenStreetMap).</strong>
                </p>
                <div id="mapHeatmap"></div>
            </div>
        </div>

        <h2 class="section-header">📊 On-Time Performance Analysis</h2>
        <div class="summary-grid">
            <div class="card">
                <h2>On-Time Delivery Percentage (by Month and Day)</h2>
                <p style="font-size: 0.9em; color: #6b7280; margin-bottom: 16px;">
                    Heatmap showing on-time delivery performance by day of week and month.
                    <span style="color: #10b981; font-weight: 600;">Green = 95%+</span>,
                    <span style="color: #dc2626; font-weight: 600;">Red = Below 85%</span>
                </p>
                <div class="ontime-heatmap" id="ontimeHeatmap">
                    <p style="text-align: center; padding: 20px;">Loading on-time performance data...</p>
                </div>
            </div>
        </div>

        <h2 class="section-header">📈 Time-Based Trends</h2>
        <div class="grid-container">
            <div class="card" style="grid-column: span 2;">
                <h2>Monthly Volume & Cost Trends</h2>
                <div class="chart-container-large"><canvas id="monthlyTrendsChart"></canvas></div>
            </div>
            <div class="card">
                <h2>Quarterly Summary</h2>
                <div class="chart-container"><canvas id="quarterlyChart"></canvas></div>
            </div>
        </div>

        <div class="grid-container">
            <div class="card" style="grid-column: span 2;">
                <h2>Monthly Movements: Completed vs Aborted</h2>
                <div class="chart-container-large"><canvas id="volumeAbortsChart"></canvas></div>
            </div>
        </div>

        <h2 class="section-header">🚚 Provider Performance Analysis</h2>
        <div class="grid-container">
            <div class="card" style="grid-column: span 2;">
                <h2>Top 10 Providers - Performance Metrics</h2>
                <div class="table-container" id="providerTable"></div>
            </div>
            <div class="card">
                <h2>Provider Efficiency Comparison</h2>
                <div class="chart-container"><canvas id="providerEfficiencyChart"></canvas></div>
            </div>
        </div>

        <h2 class="section-header">📏 Distance Band Analysis</h2>
        <div class="grid-container">
            <div class="card">
                <h2>Volume by Distance Band</h2>
                <div class="chart-container"><canvas id="distanceBandVolumeChart"></canvas></div>
            </div>
            <div class="card">
                <h2>Cost Analysis by Distance</h2>
                <div class="chart-container"><canvas id="distanceBandCPMChart"></canvas></div>
            </div>
            <div class="card">
                <h2>Distance Band Metrics</h2>
                <div class="table-container" id="distanceBandTable"></div>
            </div>
        </div>

        <h2 class="section-header">⚠️ Cost Outlier Detection</h2>
        <div class="grid-container">
            <div class="card" style="grid-column: span 2;">
                <h2>Unusual Pricing (Top 20 Outliers)</h2>
                <p style="font-size: 0.9em; color: #6b7280; margin-bottom: 10px;">
                    Movements flagged as statistical outliers using IQR method (1.5x interquartile range)
                </p>
                <div class="table-container" id="outliersTable"></div>
            </div>
        </div>

        <h2 class="section-header">💰 Most Expensive Movements</h2>
        <div class="grid-container">
            <div class="card" style="grid-column: span 2;">
                <h2>Top 10 Most Expensive by Total Cost</h2>
                <div class="table-container" id="top10CostTable"></div>
            </div>
            <div class="card" style="grid-column: span 2;">
                <h2>Top 10 Most Expensive by Cost Per Mile</h2>
                <div class="table-container" id="top10CPMTable"></div>
            </div>
        </div>

        <h2 class="section-header">💵 Budget vs Actual Performance</h2>
        <div class="grid-container">
            <div class="card">
                <h2>Budget Summary</h2>
                <div id="budgetSummary"></div>
            </div>
            <div class="card">
                <h2>Budget Accuracy</h2>
                <div class="chart-container"><canvas id="budgetAccuracyChart"></canvas></div>
            </div>
            <div class="card" style="grid-column: span 2;">
                <h2>Top 10 Over Budget Movements</h2>
                <div class="table-container" id="overBudgetTable"></div>
            </div>
        </div>

        <h2 class="section-header">👥 Internal Driver Shift Performance</h2>
        <div class="grid-container">
            <div class="card" style="grid-column: span 2;">
                <h2>Moves per Shift Analysis - By Day of Week</h2>
                <div id="dayOfWeekShiftTable"></div>
                <div class="chart-container-large" style="margin-top: 20px;"><canvas id="dayOfWeekShiftChart"></canvas></div>
            </div>
        </div>
        
        <div class="grid-container" style="margin-top: 16px;">
            <div class="card" style="grid-column: span 2;">
                <h2>Driver Performance by Shift</h2>
                <div class="table-container" id="shiftAnalysisTable"></div>
            </div>
            <div class="card">
                <h2>Average Moves per Shift Trend</h2>
                <div class="chart-container"><canvas id="shiftTrendChart"></canvas></div>
            </div>
        </div>

        <h2 class="section-header">⚡ Speed of Movement Analysis</h2>
        <div class="grid-container">
            <div class="card" style="grid-column: span 2;">
                <h2>Average Days to Complete Over Time</h2>
                <div class="chart-container-large"><canvas id="slaMonthlyChart"></canvas></div>
            </div>
            <div class="card">
                <h2>Provider Speed Comparison</h2>
                <div class="table-container" id="slaProviderTable"></div>
            </div>
        </div>

        <h2 class="section-header">🚛 Transport Type Performance</h2>
        <div class="grid-container">
            <div class="card">
                <h2>Volume by Transport Type</h2>
                <div class="chart-container"><canvas id="transportTypeVolumeChart"></canvas></div>
            </div>
            <div class="card">
                <h2>Cost Efficiency by Type</h2>
                <div class="chart-container"><canvas id="transportTypeCPMChart"></canvas></div>
            </div>
            <div class="card" style="grid-column: span 2;">
                <h2>Transport Type Comparison</h2>
                <div class="table-container" id="transportTypeTable"></div>
            </div>
        </div>

        <h2 class="section-header">🌦️ Seasonal Patterns</h2>
        <div class="grid-container">
            <div class="card" style="grid-column: span 2;">
                <h2>Monthly Activity & Cost Patterns</h2>
                <div class="chart-container-large"><canvas id="seasonalChart"></canvas></div>
            </div>
        </div>

        <h2 class="section-header">📊 Overview Charts</h2>
        <div class="grid-container">
            <div class="card">
                <h2>Movement Status</h2>
                <div class="chart-container"><canvas id="statusChart"></canvas></div>
            </div>
            <div class="card">
                <h2>Movement Type (Top 6)</h2>
                <div class="chart-container"><canvas id="movementTypeChart"></canvas></div>
            </div>
            <div class="card">
                <h2>Transport Provider (Top 5)</h2>
                <div class="chart-container"><canvas id="providerChart"></canvas></div>
            </div>
            <div class="card">
                <h2>Transport Method</h2>
                <div class="chart-container"><canvas id="methodChart"></canvas></div>
            </div>
            <div class="card text-content">
                <h2>Cost Highlights (Top 5)</h2>
                <p class="highlight">Highest Cost Per Mile:</p>
                {top_cpm_html}
                <p class="highlight">Highest Overall Cost:</p>
                {top_price_html}
            </div>
            <div class="card">
                <h2>Regional Splits (Delivery)</h2>
                <div class="chart-container"><canvas id="regionalChart"></canvas></div>
            </div>
            <div class="card">
                <h2>Stock Type</h2>
                <div class="chart-container"><canvas id="stockTypeChart"></canvas></div>
            </div>
            <div class="card list-section">
                <h2>📝 Notes & Considerations</h2>
                <ul>
                    <li><strong>Sankey Diagram:</strong> Shows cost flow through Business Area → Transport Type → Top 8 Providers.</li>
                    <li><strong>Geocoding:</strong> Uses Nominatim (OpenStreetMap) with 1 req/sec rate limit. Cached in 'geocode_cache.json'.</li>
                    <li><strong>Data Mapping:</strong> 'Final Cost' → Price, 'Miles' → Distance. CPM calculated automatically.</li>
                    <li><strong>Regional Analysis:</strong> Based on 'Business area' field (Fleet, Retail, Fleet Direct).</li>
                    <li><strong>On-Time Tracking:</strong> Uses existing 'On Time' field from source data.</li>
                    <li><strong>Date Format:</strong> DD/MM/YYYY from CSV.</li>
                    <li><strong>Distance Bands:</strong> Under 25mi, 25-50mi, 50-100mi, 100-150mi, 150+mi.</li>
                    <li><strong>Outlier Detection:</strong> IQR method with 1.5x multiplier.</li>
                    <li><strong>Efficiency Rating:</strong> 100+ = better than average CPM.</li>
                    <li><strong>Budget Analysis:</strong> Compares 'Estimated Cost' vs 'Final Cost'.</li>
                    <li><strong>Shift Analysis:</strong> Groups internal driver movements by completion date.</li>
                </ul>
            </div>
        </div>

        <footer>
            <p>Generated: {datetime.datetime.now().strftime('%d %B %Y at %H:%M')} | Powered by Jigcar Analytics</p>
        </footer>
    </div>

    <script>
        const analyticsData = {chart_map_js_data};
        const colorPalette = ['#34d399', '#10b981', '#6ee7b7', '#4b5563', '#111827', '#a7f3d0', '#d1fae5', '#059669'];
        const colorPaletteAlt = ['#10b981', '#34d399', '#6ee7b7', '#4b5563', '#a7f3d0', '#111827', '#d1fae5', '#059669'];

        function createChart(ctxId, type, data, options = {{ responsive: true, maintainAspectRatio: false }}) {{
            const ctx = document.getElementById(ctxId)?.getContext('2d');
            if (ctx) {{
                try {{
                    new Chart(ctx, {{ type, data, options }});
                }} catch (chartError) {{
                    console.error(`Error creating chart '${{ctxId}}':`, chartError);
                }}
            }} else {{
                console.error(`Canvas element with ID '${{ctxId}}' not found.`);
            }}
        }}

        if (analyticsData && Object.keys(analyticsData).length > 0) {{

            try {{
                const sankeyData = analyticsData.sankey_data || {{}};
                if (typeof Plotly !== 'undefined' && sankeyData && sankeyData.nodes && sankeyData.links && sankeyData.links.length > 0) {{
                    const nodes = sankeyData.nodes;
                    const links = sankeyData.links;
                    
                    const nodeColors = nodes.map(node => {{
                        if (['Fleet', 'Retail', 'Fleet Direct'].includes(node)) {{
                            return '#10b981';
                        }}
                        if (['Transport Provider', 'Internal Driver', '-', 'Unknown'].includes(node)) {{
                            return '#06b6d4';
                        }}
                        return '#6b7280';
                    }});
                    
                    const linkColors = links.map(link => {{
                        const sourceNode = nodes[link.source];
                        if (['Fleet', 'Retail', 'Fleet Direct'].includes(sourceNode)) {{
                            return 'rgba(16, 185, 129, 0.3)';
                        }}
                        if (['Transport Provider', 'Internal Driver'].includes(sourceNode)) {{
                            return 'rgba(6, 182, 212, 0.3)';
                        }}
                        return 'rgba(107, 114, 128, 0.3)';
                    }});
                    
                    const data = [{{
                        type: "sankey",
                        orientation: "h",
                        node: {{
                            pad: 15,
                            thickness: 20,
                            line: {{ color: "white", width: 2 }},
                            label: nodes,
                            color: nodeColors,
                            hovertemplate: '<b>%{{label}}</b><br>£%{{value:,.0f}}<extra></extra>'
                        }},
                        link: {{
                            source: links.map(l => l.source),
                            target: links.map(l => l.target),
                            value: links.map(l => l.value),
                            color: linkColors,
                            hovertemplate: '%{{source.label}} → %{{target.label}}<br>£%{{value:,.0f}}<extra></extra>'
                        }}
                    }}];

                    const layout = {{
                        font: {{ family: 'Inter, sans-serif', size: 12, color: '#4b5563' }},
                        paper_bgcolor: 'white',
                        margin: {{ l: 20, r: 20, t: 20, b: 20 }},
                        height: 600
                    }};

                    const config = {{
                        responsive: true,
                        displayModeBar: true,
                        displaylogo: false,
                        modeBarButtonsToRemove: ['lasso2d', 'select2d']
                    }};

                    Plotly.newPlot('sankeyDiagram', data, layout, config);
                }} else {{
                    document.getElementById('sankeyDiagram').innerHTML = '<p style="padding: 20px; text-align: center; color: #6b7280;">No cost flow data available.</p>';
                }}
            }} catch (sankeyError) {{
                console.error("Error creating Sankey diagram:", sankeyError);
                document.getElementById('sankeyDiagram').innerHTML = '<p style="color: #dc2626; padding: 20px;">Error creating Sankey diagram.</p>';
            }}

            function generateOntimeHeatmap() {{
                const ontimeData = analyticsData.ontime_by_day || {{}};
                
                if (!ontimeData.data || ontimeData.data.length === 0) {{
                    document.getElementById('ontimeHeatmap').innerHTML = '<p style="text-align: center; padding: 20px;">No on-time performance data available.</p>';
                    return;
                }}

                const months = ontimeData.months || [];
                const days = ontimeData.days || ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'Overall'];
                
                const dataMap = {{}};
                ontimeData.data.forEach(item => {{
                    if (!dataMap[item.period]) {{
                        dataMap[item.period] = {{}};
                    }}
                    dataMap[item.period][item.Day_of_Week] = item.percentage;
                }});

                let tableHTML = '<table><thead><tr><th>Period</th>';
                days.forEach(day => {{
                    tableHTML += `<th>${{day}}</th>`;
                }});
                tableHTML += '</tr></thead><tbody>';

                months.forEach(month => {{
                    tableHTML += `<tr><td class="month-label">${{month}}</td>`;
                    days.forEach(day => {{
                        const percentage = dataMap[month] && dataMap[month][day] ? dataMap[month][day] : null;
                        let cellClass = '';
                        let displayValue = 'N/A';
                        
                        if (percentage !== null) {{
                            displayValue = percentage.toFixed(1) + '%';
                            if (percentage >= 95) {{
                                cellClass = 'ontime-cell-high';
                            }} else if (percentage >= 85) {{
                                cellClass = 'ontime-cell-medium';
                            }} else {{
                                cellClass = 'ontime-cell-low';
                            }}
                        }}
                        
                        tableHTML += `<td class="${{cellClass}}">${{displayValue}}</td>`;
                    }});
                    tableHTML += '</tr>';
                }});

                tableHTML += '</tbody></table>';
                document.getElementById('ontimeHeatmap').innerHTML = tableHTML;
            }}

            generateOntimeHeatmap();

            const monthlyData = analyticsData.time_trends?.monthly || [];
            if (monthlyData.length > 0) {{
                createChart('monthlyTrendsChart', 'line', {{
                    labels: monthlyData.map(d => d.period),
                    datasets: [
                        {{ label: 'Volume', data: monthlyData.map(d => d.volume || 0), borderColor: colorPalette[0], backgroundColor: 'rgba(52, 211, 153, 0.1)', yAxisID: 'y', tension: 0.3, fill: true }},
                        {{ label: 'Total Cost (£)', data: monthlyData.map(d => d.total_cost || 0), borderColor: colorPalette[1], backgroundColor: 'rgba(16, 185, 129, 0.1)', yAxisID: 'y1', tension: 0.3, fill: true }}
                    ]
                }}, {{
                    responsive: true, maintainAspectRatio: false,
                    scales: {{
                        y: {{ beginAtZero: true, position: 'left', title: {{ display: true, text: 'Volume' }} }},
                        y1: {{ beginAtZero: true, position: 'right', title: {{ display: true, text: 'Cost (£)' }}, grid: {{ drawOnChartArea: false }} }}
                    }},
                    plugins: {{ legend: {{ position: 'top' }} }}
                }});
            }}

            const quarterlyData = analyticsData.time_trends?.quarterly || [];
            if (quarterlyData.length > 0) {{
                createChart('quarterlyChart', 'bar', {{
                    labels: quarterlyData.map(d => d.period),
                    datasets: [{{ label: 'Volume', data: quarterlyData.map(d => d.volume), backgroundColor: colorPalette[0] }}]
                }}, {{ responsive: true, maintainAspectRatio: false, scales: {{ y: {{ beginAtZero: true }} }}, plugins: {{ legend: {{ display: false }} }} }});
            }}

            const volumeAbortsData = analyticsData.volume_aborts_monthly || [];
            if (volumeAbortsData.length > 0) {{
                createChart('volumeAbortsChart', 'bar', {{
                    labels: volumeAbortsData.map(d => d.period),
                    datasets: [
                        {{ label: 'Completed Movements', data: volumeAbortsData.map(d => d.movements), backgroundColor: colorPalette[0], stack: 'Stack 0' }},
                        {{ label: 'Aborted Movements', data: volumeAbortsData.map(d => d.aborts), backgroundColor: '#dc2626', stack: 'Stack 0' }},
                        {{
                            label: 'Abort Rate %',
                            data: volumeAbortsData.map(d => d.abort_pct),
                            type: 'line',
                            borderColor: '#f59e0b',
                            backgroundColor: 'transparent',
                            yAxisID: 'y1',
                            tension: 0.3
                        }}
                    ]
                }}, {{
                    responsive: true, maintainAspectRatio: false,
                    scales: {{
                        y: {{ beginAtZero: true, stacked: true, position: 'left', title: {{ display: true, text: 'Movements' }} }},
                        y1: {{ beginAtZero: true, position: 'right', title: {{ display: true, text: 'Abort Rate (%)' }}, grid: {{ drawOnChartArea: false }} }}
                    }},
                    plugins: {{ legend: {{ position: 'top' }} }}
                }});
            }}

            const providerPerf = analyticsData.provider_performance || [];
            if (providerPerf.length > 0) {{
                let tableHTML = '<table><thead><tr><th>Provider</th><th>Moves</th><th>Avg Cost</th><th>Avg CPM</th><th>On-Time %</th><th>Efficiency</th></tr></thead><tbody>';
                providerPerf.forEach(p => {{
                    const efficiency = p.efficiency_rating ? p.efficiency_rating.toFixed(0) + '%' : 'N/A';
                    const efficiencyBadge = p.efficiency_rating >= 100 ? 'badge-low' : (p.efficiency_rating >= 80 ? 'badge-medium' : 'badge-high');
                    const ontime = p.ontime_pct ? p.ontime_pct.toFixed(1) + '%' : 'N/A';
                    tableHTML += `<tr>
                        <td>${{p['Driver / Provider'] || 'N/A'}}</td>
                        <td>${{p.total_moves || 0}}</td>
                        <td>£${{p.avg_cost ? p.avg_cost.toFixed(2) : 'N/A'}}</td>
                        <td>£${{p.avg_cpm ? p.avg_cpm.toFixed(2) : 'N/A'}}</td>
                        <td>${{ontime}}</td>
                        <td><span class="badge ${{efficiencyBadge}}">${{efficiency}}</span></td>
                    </tr>`;
                }});
                tableHTML += '</tbody></table>';
                document.getElementById('providerTable').innerHTML = tableHTML;
            }}

            if (providerPerf.length > 0) {{
                const topProviders = providerPerf.slice(0, 8);
                createChart('providerEfficiencyChart', 'bar', {{
                    labels: topProviders.map(p => p['Driver / Provider']),
                    datasets: [{{
                        label: 'Efficiency Rating (%)',
                        data: topProviders.map(p => p.efficiency_rating || 0),
                        backgroundColor: topProviders.map(p => p.efficiency_rating >= 100 ? colorPalette[2] : colorPalette[4])
                    }}]
                }}, {{
                    indexAxis: 'y', responsive: true, maintainAspectRatio: false,
                    scales: {{ x: {{ beginAtZero: true }} }},
                    plugins: {{ legend: {{ display: false }} }}
                }});
            }}

            const distanceData = analyticsData.distance_analysis || [];
            if (distanceData.length > 0) {{
                createChart('distanceBandVolumeChart', 'doughnut', {{
                    labels: distanceData.map(d => d.Distance_Band),
                    datasets: [{{ data: distanceData.map(d => d.volume), backgroundColor: colorPaletteAlt }}]
                }}, {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ position: 'bottom' }} }} }});

                const bandLabels = distanceData.map(d => d.Distance_Band);
                const avgCosts = distanceData.map(d => d.avg_cost || 0);
                const volumes = distanceData.map(d => d.volume || 0);
                const totalMoves = volumes.reduce((sum, v) => sum + v, 0);
                const volumePercentages = volumes.map(v => (v / totalMoves * 100));
                const overallAvgCost = avgCosts.reduce((sum, cost, idx) => sum + (cost * volumes[idx]), 0) / totalMoves;
                const avgLine = new Array(avgCosts.length).fill(overallAvgCost);
                
                createChart('distanceBandCPMChart', 'bar', {{
                    labels: bandLabels,
                    datasets: [
                        {{ type: 'bar', label: 'Avg Cost per Car', data: avgCosts, backgroundColor: '#10b981', yAxisID: 'y' }},
                        {{ type: 'line', label: 'Volume %', data: volumePercentages, borderColor: '#06b6d4', backgroundColor: 'transparent', borderWidth: 3, pointRadius: 6, yAxisID: 'y1', tension: 0.4 }},
                        {{ type: 'line', label: 'Average Cost', data: avgLine, borderColor: '#111827', backgroundColor: 'transparent', borderWidth: 2, borderDash: [8, 4], pointRadius: 0, yAxisID: 'y' }}
                    ]
                }}, {{
                    responsive: true, maintainAspectRatio: false,
                    scales: {{
                        y: {{ beginAtZero: true, position: 'left', title: {{ display: true, text: 'Avg Cost (£)' }} }},
                        y1: {{ beginAtZero: true, position: 'right', title: {{ display: true, text: 'Volume %' }}, grid: {{ drawOnChartArea: false }} }}
                    }},
                    plugins: {{ legend: {{ position: 'top' }} }}
                }});

                let distTableHTML = '<table><thead><tr><th>Band</th><th>Volume</th><th>Avg Distance</th><th>Avg Cost</th><th>Avg CPM</th></tr></thead><tbody>';
                distanceData.forEach(d => {{
                    distTableHTML += `<tr>
                        <td>${{d.Distance_Band}}</td>
                        <td>${{d.volume}}</td>
                        <td>${{d.avg_distance ? d.avg_distance.toFixed(1) : 'N/A'}} mi</td>
                        <td>£${{d.avg_cost ? d.avg_cost.toFixed(2) : 'N/A'}}</td>
                        <td>£${{d.avg_cpm ? d.avg_cpm.toFixed(2) : 'N/A'}}</td>
                    </tr>`;
                }});
                distTableHTML += '</tbody></table>';
                document.getElementById('distanceBandTable').innerHTML = distTableHTML;
            }}

            const outliers = analyticsData.outliers || [];
            if (outliers.length > 0) {{
                let outlierHTML = '<table><thead><tr><th>Postcode</th><th>Distance</th><th>Price</th><th>CPM</th><th>Provider</th><th>Type</th></tr></thead><tbody>';
                outliers.forEach(o => {{
                    const badge = o.outlier_type === 'Both' ? 'badge-high' : 'badge-medium';
                    outlierHTML += `<tr>
                        <td>${{o['Delivery Postcode'] || 'N/A'}}</td>
                        <td>${{o.Distance_Num ? o.Distance_Num.toFixed(1) : 'N/A'}} mi</td>
                        <td>£${{o.Price_Num ? o.Price_Num.toFixed(2) : 'N/A'}}</td>
                        <td>£${{o.CPM_Num ? o.CPM_Num.toFixed(2) : 'N/A'}}</td>
                        <td>${{o['Driver / Provider'] || 'N/A'}}</td>
                        <td><span class="badge ${{badge}}">${{o.outlier_type}}</span></td>
                    </tr>`;
                }});
                outlierHTML += '</tbody></table>';
                document.getElementById('outliersTable').innerHTML = outlierHTML;
            }} else {{
                document.getElementById('outliersTable').innerHTML = '<p style="padding: 20px; text-align: center;">No outliers detected.</p>';
            }}

            const top10Cost = analyticsData.top_10_by_cost || [];
            if (top10Cost.length > 0) {{
                let costTableHTML = '<table><thead><tr><th>VRM</th><th>Postcode</th><th>Distance</th><th>Total Cost</th><th>CPM</th><th>Provider</th><th>Type</th><th>Date</th></tr></thead><tbody>';
                top10Cost.forEach((m, idx) => {{
                    const date = m['Completed On'] ? new Date(m['Completed On']).toLocaleDateString('en-GB') : 'N/A';
                    const rankBadge = idx < 3 ? 'badge-high' : (idx < 6 ? 'badge-medium' : 'badge-low');
                    costTableHTML += `<tr>
                        <td><span class="badge ${{rankBadge}}">#${{idx + 1}}</span> ${{m.VRM || 'N/A'}}</td>
                        <td>${{m['Delivery Postcode'] || 'N/A'}}</td>
                        <td>${{m.Distance_Num ? m.Distance_Num.toFixed(1) : 'N/A'}} mi</td>
                        <td><strong>£${{m.Price_Num ? m.Price_Num.toFixed(2) : 'N/A'}}</strong></td>
                        <td>£${{m.CPM_Num ? m.CPM_Num.toFixed(2) : 'N/A'}}</td>
                        <td>${{m['Driver / Provider'] || 'N/A'}}</td>
                        <td>${{m['Movement Type'] || 'N/A'}}</td>
                        <td>${{date}}</td>
                    </tr>`;
                }});
                costTableHTML += '</tbody></table>';
                document.getElementById('top10CostTable').innerHTML = costTableHTML;
            }}

            const top10CPM = analyticsData.top_10_by_cpm || [];
            if (top10CPM.length > 0) {{
                let cpmTableHTML = '<table><thead><tr><th>VRM</th><th>Postcode</th><th>Distance</th><th>Total Cost</th><th>CPM</th><th>Provider</th><th>Type</th><th>Date</th></tr></thead><tbody>';
                top10CPM.forEach((m, idx) => {{
                    const date = m['Completed On'] ? new Date(m['Completed On']).toLocaleDateString('en-GB') : 'N/A';
                    const rankBadge = idx < 3 ? 'badge-high' : (idx < 6 ? 'badge-medium' : 'badge-low');
                    cpmTableHTML += `<tr>
                        <td><span class="badge ${{rankBadge}}">#${{idx + 1}}</span> ${{m.VRM || 'N/A'}}</td>
                        <td>${{m['Delivery Postcode'] || 'N/A'}}</td>
                        <td>${{m.Distance_Num ? m.Distance_Num.toFixed(1) : 'N/A'}} mi</td>
                        <td>£${{m.Price_Num ? m.Price_Num.toFixed(2) : 'N/A'}}</td>
                        <td><strong>£${{m.CPM_Num ? m.CPM_Num.toFixed(2) : 'N/A'}}</strong></td>
                        <td>${{m['Driver / Provider'] || 'N/A'}}</td>
                        <td>${{m['Movement Type'] || 'N/A'}}</td>
                        <td>${{date}}</td>
                    </tr>`;
                }});
                cpmTableHTML += '</tbody></table>';
                document.getElementById('top10CPMTable').innerHTML = cpmTableHTML;
            }}

            const budgetData = analyticsData.budget_analysis || {{}};
            if (budgetData && Object.keys(budgetData).length > 0 && budgetData.records_analyzed > 0) {{
                const budgetSummaryHTML = `
                    <div class="kpi-grid">
                        <div class="kpi"><span class="value">£${{budgetData.total_estimated ? budgetData.total_estimated.toFixed(0) : '0'}}</span><span class="label">Total Estimated</span></div>
                        <div class="kpi"><span class="value">£${{budgetData.total_actual ? budgetData.total_actual.toFixed(0) : '0'}}</span><span class="label">Total Actual</span></div>
                        <div class="kpi"><span class="value" style="color: ${{budgetData.total_variance >= 0 ? '#dc2626' : '#10b981'}}">£${{budgetData.total_variance ? budgetData.total_variance.toFixed(0) : '0'}}</span><span class="label">Variance</span></div>
                        <div class="kpi"><span class="value">${{budgetData.avg_variance_pct ? budgetData.avg_variance_pct.toFixed(1) : '0'}}%</span><span class="label">Avg Variance %</span></div>
                    </div>
                `;
                document.getElementById('budgetSummary').innerHTML = budgetSummaryHTML;

                createChart('budgetAccuracyChart', 'doughnut', {{
                    labels: ['Over Budget', 'Under Budget', 'On Budget'],
                    datasets: [{{ data: [budgetData.over_budget_count || 0, budgetData.under_budget_count || 0, budgetData.on_budget_count || 0], backgroundColor: ['#dc2626', '#10b981', '#f59e0b'] }}]
                }}, {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ position: 'bottom' }} }} }});

                const overBudgetData = budgetData.top_over_budget || [];
                if (overBudgetData.length > 0) {{
                    let overBudgetHTML = '<table><thead><tr><th>VRM</th><th>Postcode</th><th>Estimated</th><th>Actual</th><th>Variance £</th><th>Variance %</th><th>Provider</th></tr></thead><tbody>';
                    overBudgetData.forEach((m, idx) => {{
                        const badge = idx < 3 ? 'badge-high' : (idx < 6 ? 'badge-medium' : 'badge-low');
                        overBudgetHTML += `<tr>
                            <td><span class="badge ${{badge}}">#${{idx + 1}}</span> ${{m.VRM || 'N/A'}}</td>
                            <td>${{m['Delivery Postcode'] || 'N/A'}}</td>
                            <td>£${{m.Estimated_Cost_Num ? m.Estimated_Cost_Num.toFixed(2) : 'N/A'}}</td>
                            <td>£${{m.Price_Num ? m.Price_Num.toFixed(2) : 'N/A'}}</td>
                            <td><strong style="color: #dc2626;">+£${{m.Cost_Variance ? m.Cost_Variance.toFixed(2) : 'N/A'}}</strong></td>
                            <td>+${{m.Cost_Variance_Pct ? m.Cost_Variance_Pct.toFixed(1) : 'N/A'}}%</td>
                            <td>${{m['Driver / Provider'] || 'N/A'}}</td>
                        </tr>`;
                    }});
                    overBudgetHTML += '</tbody></table>';
                    document.getElementById('overBudgetTable').innerHTML = overBudgetHTML;
                }}
            }} else {{
                document.getElementById('budgetSummary').innerHTML = '<p style="padding: 20px; text-align: center;">No budget data available.</p>';
            }}

            const shiftData = analyticsData.shift_analysis || {{}};
            
            if (shiftData && shiftData.day_of_week_analysis && shiftData.day_of_week_analysis.length > 0) {{
                const dayData = shiftData.day_of_week_analysis;
                
                let dayTableHTML = `
                    <div style="margin-bottom: 20px;">
                        <table style="width: 100%; border-collapse: collapse;">
                            <thead>
                                <tr style="background-color: var(--jigcar-black); color: white;">
                                    <th style="padding: 12px; text-align: left; border: 1px solid #ddd;"></th>`;
                
                dayData.forEach(d => {{
                    dayTableHTML += `<th style="padding: 12px; text-align: center; border: 1px solid #ddd;">${{d.Day_of_Week}}</th>`;
                }});
                
                dayTableHTML += `</tr></thead><tbody>`;
                
                dayTableHTML += `<tr><td style="padding: 12px; font-weight: 600; background-color: var(--jigcar-gray-light); border: 1px solid #ddd;">Resource (Shifts)</td>`;
                dayData.forEach(d => {{
                    dayTableHTML += `<td style="padding: 12px; text-align: center; border: 1px solid #ddd;">${{d.total_shifts || 0}}</td>`;
                }});
                dayTableHTML += `</tr>`;
                
                dayTableHTML += `<tr><td style="padding: 12px; font-weight: 600; background-color: var(--jigcar-gray-light); border: 1px solid #ddd;">Cars Moved</td>`;
                dayData.forEach(d => {{
                    dayTableHTML += `<td style="padding: 12px; text-align: center; border: 1px solid #ddd;">${{d.total_cars_moved || 0}}</td>`;
                }});
                dayTableHTML += `</tr>`;
                
                dayTableHTML += `<tr><td style="padding: 12px; font-weight: 600; background-color: var(--jigcar-gray-light); border: 1px solid #ddd;">Moves per Shift</td>`;
                dayData.forEach(d => {{
                    const movesPerShift = d.avg_moves_per_shift ? d.avg_moves_per_shift.toFixed(1) : '0.0';
                    dayTableHTML += `<td style="padding: 12px; text-align: center; font-weight: 700; border: 1px solid #ddd;">${{movesPerShift}}</td>`;
                }});
                dayTableHTML += `</tr>`;
                
                dayTableHTML += `</tbody></table></div>`;
                document.getElementById('dayOfWeekShiftTable').innerHTML = dayTableHTML;
                
                createChart('dayOfWeekShiftChart', 'bar', {{
                    labels: dayData.map(d => d.Day_of_Week),
                    datasets: [{{
                        label: 'Movements per Shift',
                        data: dayData.map(d => d.avg_moves_per_shift || 0),
                        backgroundColor: '#4A9FF5',
                        borderColor: '#4A9FF5',
                        borderWidth: 1
                    }}]
                }}, {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            title: {{ display: true, text: 'Movements per shift', font: {{ size: 14 }} }},
                            grid: {{ color: '#e0e0e0' }}
                        }},
                        x: {{
                            grid: {{ display: false }}
                        }}
                    }},
                    plugins: {{
                        legend: {{ display: false }},
                        title: {{
                            display: true,
                            text: 'Movements per shift',
                            align: 'start',
                            font: {{ size: 16, weight: 'normal' }},
                            color: '#666',
                            padding: {{ bottom: 20 }}
                        }}
                    }}
                }});
            }}
            
            if (shiftData && shiftData.driver_performance && shiftData.driver_performance.length > 0) {{
                let shiftTableHTML = '<table><thead><tr><th>Driver</th><th>Total Shifts</th><th>Avg Moves/Shift</th><th>Total Moves</th><th>Avg Cost/Shift</th><th>Avg Miles/Shift</th></tr></thead><tbody>';
                shiftData.driver_performance.forEach(d => {{
                    shiftTableHTML += `<tr>
                        <td>${{d['Driver / Provider'] || 'N/A'}}</td>
                        <td>${{d.total_shifts || 0}}</td>
                        <td><strong>${{d.avg_moves_per_shift ? d.avg_moves_per_shift.toFixed(1) : 'N/A'}}</strong></td>
                        <td>${{d.total_moves || 0}}</td>
                        <td>£${{d.avg_cost_per_shift ? d.avg_cost_per_shift.toFixed(2) : 'N/A'}}</td>
                        <td>${{d.avg_miles_per_shift ? d.avg_miles_per_shift.toFixed(1) : 'N/A'}}</td>
                    </tr>`;
                }});
                shiftTableHTML += '</tbody></table>';
                document.getElementById('shiftAnalysisTable').innerHTML = shiftTableHTML;

                const shiftTrend = shiftData.monthly_trend || [];
                if (shiftTrend.length > 0) {{
                    createChart('shiftTrendChart', 'line', {{
                        labels: shiftTrend.map(d => d.period),
                        datasets: [{{
                            label: 'Avg Moves per Shift',
                            data: shiftTrend.map(d => d.avg_moves_per_shift),
                            borderColor: colorPalette[0],
                            backgroundColor: 'rgba(52, 211, 153, 0.1)',
                            tension: 0.3,
                            fill: true
                        }}]
                    }}, {{ responsive: true, maintainAspectRatio: false, scales: {{ y: {{ beginAtZero: true }} }}, plugins: {{ legend: {{ display: false }} }} }});
                }}
            }}

            const speedData = analyticsData.speed_analysis || {{}};
            if (speedData && speedData.monthly_overall && speedData.monthly_overall.length > 0) {{
                createChart('slaMonthlyChart', 'line', {{
                    labels: speedData.monthly_overall.map(d => d.period),
                    datasets: [{{
                        label: 'Average Days to Complete',
                        data: speedData.monthly_overall.map(d => d.avg_days),
                        borderColor: colorPalette[1],
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        tension: 0.3,
                        fill: true
                    }}]
                }}, {{ responsive: true, maintainAspectRatio: false, scales: {{ y: {{ beginAtZero: true }} }}, plugins: {{ legend: {{ display: false }} }} }});

                const providerAvgs = speedData.provider_averages || [];
                if (providerAvgs.length > 0) {{
                    let slaTableHTML = '<table><thead><tr><th>Provider</th><th>Avg Days</th><th>Volume</th></tr></thead><tbody>';
                    providerAvgs.forEach(p => {{
                        slaTableHTML += `<tr>
                            <td>${{p['Driver / Provider'] || 'N/A'}}</td>
                            <td><strong>${{p.avg_days ? p.avg_days.toFixed(1) : 'N/A'}}</strong> days</td>
                            <td>${{p.volume || 0}}</td>
                        </tr>`;
                    }});
                    slaTableHTML += '</tbody></table>';
                    document.getElementById('slaProviderTable').innerHTML = slaTableHTML;
                }}
            }}

            const transportTypeData = analyticsData.transport_type_analysis || [];
            if (transportTypeData.length > 0) {{
                createChart('transportTypeVolumeChart', 'bar', {{
                    labels: transportTypeData.map(t => t['Transport Type']),
                    datasets: [{{ label: 'Volume', data: transportTypeData.map(t => t.volume), backgroundColor: colorPaletteAlt }}]
                }}, {{ responsive: true, maintainAspectRatio: false, scales: {{ y: {{ beginAtZero: true }} }}, plugins: {{ legend: {{ display: false }} }} }});

                createChart('transportTypeCPMChart', 'bar', {{
                    labels: transportTypeData.map(t => t['Transport Type']),
                    datasets: [{{ label: 'Avg CPM', data: transportTypeData.map(t => t.avg_cpm), backgroundColor: colorPalette[1] }}]
                }}, {{ responsive: true, maintainAspectRatio: false, scales: {{ y: {{ beginAtZero: true }} }}, plugins: {{ legend: {{ display: false }} }} }});

                let transportTableHTML = '<table><thead><tr><th>Type</th><th>Volume</th><th>Total Cost</th><th>Avg Cost</th><th>Avg CPM</th><th>Avg Distance</th><th>On-Time %</th></tr></thead><tbody>';
                transportTypeData.forEach(t => {{
                    const ontime = t.ontime_pct ? t.ontime_pct.toFixed(1) + '%' : 'N/A';
                    transportTableHTML += `<tr>
                        <td><strong>${{t['Transport Type'] || 'N/A'}}</strong></td>
                        <td>${{t.volume || 0}}</td>
                        <td>£${{t.total_cost ? t.total_cost.toFixed(2) : 'N/A'}}</td>
                        <td>£${{t.avg_cost ? t.avg_cost.toFixed(2) : 'N/A'}}</td>
                        <td>£${{t.avg_cpm ? t.avg_cpm.toFixed(2) : 'N/A'}}</td>
                        <td>${{t.avg_distance ? t.avg_distance.toFixed(1) : 'N/A'}} mi</td>
                        <td>${{ontime}}</td>
                    </tr>`;
                }});
                transportTableHTML += '</tbody></table>';
                document.getElementById('transportTypeTable').innerHTML = transportTableHTML;
            }}

            const seasonalData = analyticsData.seasonal_patterns || [];
            if (seasonalData.length > 0) {{
                createChart('seasonalChart', 'line', {{
                    labels: seasonalData.map(d => d.Month_Name),
                    datasets: [
                        {{ label: 'Avg Volume', data: seasonalData.map(d => d.avg_volume), borderColor: colorPalette[0], backgroundColor: 'rgba(52, 211, 153, 0.1)', yAxisID: 'y', tension: 0.3, fill: true }},
                        {{ label: 'Avg Cost (£)', data: seasonalData.map(d => d.avg_cost), borderColor: colorPalette[1], backgroundColor: 'rgba(16, 185, 129, 0.1)', yAxisID: 'y1', tension: 0.3, fill: true }}
                    ]
                }}, {{
                    responsive: true, maintainAspectRatio: false,
                    scales: {{
                        y: {{ beginAtZero: true, position: 'left', title: {{ display: true, text: 'Avg Volume' }} }},
                        y1: {{ beginAtZero: true, position: 'right', title: {{ display: true, text: 'Avg Cost (£)' }}, grid: {{ drawOnChartArea: false }} }}
                    }},
                    plugins: {{ legend: {{ position: 'top' }} }}
                }});
            }}

            const statusData = analyticsData.status_counts || {{}};
            if (Object.keys(statusData).length > 0) {{
                createChart('statusChart', 'pie', {{
                    labels: Object.keys(statusData),
                    datasets: [{{ data: Object.values(statusData), backgroundColor: colorPalette }}]
                }}, {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ position: 'bottom' }} }} }});
            }}

            const moveTypeData = analyticsData.movement_type_counts || {{}};
            if (Object.keys(moveTypeData).length > 0) {{
                createChart('movementTypeChart', 'doughnut', {{
                    labels: Object.keys(moveTypeData),
                    datasets: [{{ data: Object.values(moveTypeData), backgroundColor: colorPaletteAlt }}]
                }}, {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ position: 'bottom' }} }} }});
            }}

            const providerData = analyticsData.provider_counts || {{}};
            if (Object.keys(providerData).length > 0) {{
                createChart('providerChart', 'bar', {{
                    labels: Object.keys(providerData),
                    datasets: [{{ label: '# Movements', data: Object.values(providerData), backgroundColor: colorPalette[0] }}]
                }}, {{ indexAxis: 'y', responsive: true, maintainAspectRatio: false, scales: {{ x: {{ beginAtZero: true }} }}, plugins: {{ legend: {{ display: false }} }} }});
            }}

            const methodData = analyticsData.method_counts || {{}};
            if (Object.keys(methodData).length > 0) {{
                createChart('methodChart', 'pie', {{
                    labels: Object.keys(methodData),
                    datasets: [{{ data: Object.values(methodData), backgroundColor: [colorPalette[1], colorPalette[0]] }}]
                }}, {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ position: 'bottom' }} }} }});
            }}

            const stockData = analyticsData.stock_counts || {{}};
            if (Object.keys(stockData).length > 0) {{
                createChart('stockTypeChart', 'doughnut', {{
                    labels: Object.keys(stockData),
                    datasets: [{{ data: Object.values(stockData), backgroundColor: [colorPalette[2], colorPalette[3]] }}]
                }}, {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ position: 'bottom' }} }} }});
            }}

            const regionalRawData = analyticsData.regional_data || [];
            if (regionalRawData.length > 0) {{
                const regionColName = analyticsData.region_column_name || 'Region';
                const regionalLabels = regionalRawData.map(item => item[regionColName] || 'Unknown');
                const regionalMoves = regionalRawData.map(item => item.moves || 0);
                const regionalAvgCost = regionalRawData.map(item => item.avg_cost);
                const regionalAvgSla = regionalRawData.map(item => item.avg_sla);

                createChart('regionalChart', 'bar', {{
                    labels: regionalLabels,
                    datasets: [{{ label: '# Moves', data: regionalMoves, backgroundColor: colorPalette[0] }}]
                }}, {{
                    indexAxis: 'x', responsive: true, maintainAspectRatio: false,
                    scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: '# of Moves' }} }} }},
                    plugins: {{
                        legend: {{ display: false }},
                        tooltip: {{
                            callbacks: {{
                                footer: function(tooltipItems) {{
                                    let index = tooltipItems[0].dataIndex;
                                    let cost = regionalAvgCost[index] ? `£${{regionalAvgCost[index].toFixed(2)}}` : 'N/A';
                                    let sla = regionalAvgSla[index] ? `${{regionalAvgSla[index].toFixed(1)}} days` : 'N/A';
                                    return `Avg Cost: ${{cost}}\\nAvg SLA: ${{sla}}`;
                                }}
                            }}
                        }}
                    }}
                }});
            }}
        }}

        document.addEventListener('DOMContentLoaded', function () {{
            const heatMapPoints = analyticsData?.heatmap_points || [];
            const mapElement = document.getElementById('mapHeatmap');

            if(mapElement && typeof L !== 'undefined' && heatMapPoints.length > 0) {{
                const validHeatPoints = heatMapPoints.filter(p => p && p.length === 3 && p[0] !== null && !isNaN(p[0]) && p[1] !== null && !isNaN(p[1]) && p[2] !== null && !isNaN(p[2]));

                if (validHeatPoints.length > 0) {{
                    const prices = validHeatPoints.map(p => p[2]);
                    const maxPriceInData = prices.length > 0 ? Math.max(...prices) : 500;

                    const heatOptions = {{
                        radius: 25,
                        blur: 15,
                        maxZoom: 14,
                        max: maxPriceInData * 0.85,
                        minOpacity: 0.2,
                        gradient: {{ 0.2: 'blue', 0.4: 'cyan', 0.6: 'lime', 0.8: 'yellow', 0.9: 'orange', 1.0: 'red' }}
                    }};

                    const map = L.map('mapHeatmap').setView([54.5, -3.0], 6);

                    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                        attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                    }}).addTo(map);

                    try {{
                        L.heatLayer(validHeatPoints, heatOptions).addTo(map);
                    }} catch (e) {{
                        console.error("Error creating Leaflet heat layer:", e);
                        mapElement.innerHTML = "<p style='color: red; padding: 10px;'>Error initializing heatmap.</p>";
                    }}
                }} else {{
                    mapElement.innerHTML = "<p style='padding: 10px;'>No valid points for heatmap.</p>";
                }}
            }} else if (mapElement) {{
                mapElement.innerHTML = "<p style='padding: 10px;'>No heatmap data available.</p>";
            }}
        }});
    </script>
</body>
</html>
"""

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"✓ Successfully generated comprehensive infographic: {output_file}\n")
    except IOError as e:
        print(f"✗ Error writing HTML file: {e}\n")

def save_cache(cache_data, filename):
    print(f"Saving {len(cache_data)} items to geocode cache: {filename}")
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=4)
        print("✓ Cache saved successfully.\n")
    except IOError as e:
        print(f"✗ Error saving geocode cache: {e}\n")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("COMPREHENSIVE PERRYS TRANSPORT ANALYTICS GENERATOR")
    print("="*70 + "\n")
    
    print("📅 DATE FILTER SETTINGS:")
    if FILTER_START_DATE or FILTER_END_DATE:
        print(f"  Start Date: {FILTER_START_DATE or 'No limit'}")
        print(f"  End Date:   {FILTER_END_DATE or 'No limit'}")
        print(f"  ⚠️  Date filtering is ACTIVE")
    else:
        print(f"  ✓ No date filtering - analyzing ALL data")
    print()

    dataframe = fetch_data(GOOGLE_SHEET_CSV_URL)
    
    if dataframe is not None:
        dataframe = preprocess_and_geocode(dataframe)
        
        if dataframe is not None:
            all_analytics = calculate_analytics(dataframe)
            generate_html_infographic(all_analytics, OUTPUT_HTML_FILE)
        else:
            print("✗ Skipping analytics due to preprocessing errors.\n")
    else:
        print("✗ Skipping due to data fetching errors.\n")

    save_cache(geocode_cache, GEOCODE_CACHE_FILE)
