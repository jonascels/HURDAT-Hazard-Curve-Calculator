import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from scipy.stats import genextreme
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# --- CONFIGURATION ---
# ==========================================
st.set_page_config(page_title="Hurricane Hazard Analysis", layout="wide")

LOCAL_FILENAME = "hurdat2-1851-2024-040425.txt"
STANDARD_RPS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

# ==========================================
# --- 1. DATA LOADING & CACHING ---
# ==========================================
@st.cache_data(show_spinner="Loading and parsing hurricane data...")
def load_and_parse_data():
    def get_category(knots):
        if knots < 34: return 'TD'
        if knots < 64: return 'TS'
        if knots < 83: return 'Cat 1'
        if knots < 96: return 'Cat 2'
        if knots < 113: return 'Cat 3'
        if knots < 137: return 'Cat 4'
        return 'Cat 5'

    if not os.path.exists(LOCAL_FILENAME): 
        return pd.DataFrame()
    
    with open(LOCAL_FILENAME, 'r') as f: 
        lines = f.readlines()
        
    data = []
    current_id, current_name, current_year = None, None, None
    
    for line in lines:
        parts = [p.strip() for p in line.split(',')]
        if len(parts) == 4: 
            current_id, current_name = parts[0], parts[1]
            current_year = int(current_id[4:]) 
        elif len(parts) >= 20:
            lat = float(parts[4][:-1]) * (-1 if parts[4].endswith('S') else 1)
            lon = float(parts[5][:-1]) * (-1 if parts[5].endswith('W') else 1)
            try: wind = int(parts[6])
            except: wind = 0
            
            wind_kmh = wind * 1.852
            
            data.append({
                'StormID': current_id, 'Name': current_name, 'Year': current_year,
                'Date': parts[0], 'Time': parts[1],
                'Lat': lat, 'Lon': lon, 
                'WindSpeed_Knots': wind, 'WindSpeed_KMH': wind_kmh,
                'Category': get_category(wind)
            })
    return pd.DataFrame(data)

# ==========================================
# --- 2. MATH & STATS FUNCTIONS ---
# ==========================================
def haversine_distance_vectorized(lat1_deg, lon1_deg, lat2_deg, lon2_deg):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1_deg, lon1_deg, lat2_deg, lon2_deg])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * 6371

def generate_circle_points(lat, lon, km):
    lats, lons = [], []
    R = 6371
    lat1 = np.radians(lat)
    lon1 = np.radians(lon)
    for bearing in range(0, 361, 5):
        brng = np.radians(bearing)
        d = km
        lat2 = np.arcsin(np.sin(lat1)*np.cos(d/R) + np.cos(lat1)*np.sin(d/R)*np.cos(brng))
        lon2 = lon1 + np.arctan2(np.sin(brng)*np.sin(d/R)*np.cos(lat1), np.cos(d/R)-np.sin(lat1)*np.sin(lat2))
        lats.append(np.degrees(lat2))
        lons.append(np.degrees(lon2))
    return lats, lons

def get_empirical_raw(storm_data, total_years):
    winds = sorted([d['Wind'] for d in storm_data], reverse=True)
    if not winds: return [], []
    unique_winds = sorted(list(set(winds)))
    x_rp, y_wind = [], []
    for v in unique_winds:
        count = len([w for w in winds if w >= v])
        rate = count / total_years
        if rate > 0:
            x_rp.append(1.0 / rate)
            y_wind.append(v)
    return x_rp, y_wind

def get_empirical_interpolated(storm_data, total_years, target_rps):
    x_raw, y_raw = get_empirical_raw(storm_data, total_years)
    if not x_raw: return [0]*len(target_rps)
    sorted_pairs = sorted(zip(x_raw, y_raw))
    x_sorted = [p[0] for p in sorted_pairs]
    y_sorted = [p[1] for p in sorted_pairs]
    log_x = np.log(x_sorted)
    log_targets = np.log(target_rps)
    interp_winds = np.interp(log_targets, log_x, y_sorted, left=0, right=y_sorted[-1])
    return interp_winds

def get_gev_curve(storm_data, total_years, target_rps_array, max_cap):
    winds = [d['Wind'] for d in storm_data]
    if len(winds) < 5: return [0]*len(target_rps_array)
    lambda_rate = len(winds) / total_years
    try:
        shape_c, loc, scale = genextreme.fit(winds)
    except:
        return [0]*len(target_rps_array)
    fitted_winds = []
    for rp in target_rps_array:
        target_prob_exceed = 1.0 / (lambda_rate * rp)
        if target_prob_exceed >= 1:
            fitted_winds.append(0)
            continue
        target_cdf = 1 - target_prob_exceed
        w = genextreme.ppf(target_cdf, shape_c, loc, scale)
        if w > max_cap: w = max_cap
        if w < 0: w = 0
        fitted_winds.append(w)
    return fitted_winds

def generate_excel_bytes(summary, events, curve_emp, curve_gev):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        pd.DataFrame(summary).to_excel(writer, sheet_name="Summary_Interpolated", index=False)
        pd.DataFrame(events).to_excel(writer, sheet_name="Raw_Events", index=False)
        pd.DataFrame(curve_emp).to_excel(writer, sheet_name="Curve_Data_Empirical", index=False)
        pd.DataFrame(curve_gev).to_excel(writer, sheet_name="Curve_Data_GEV", index=False)
    return output.getvalue()

# ==========================================
# --- 3. STREAMLIT APP LAYOUT & LOGIC ---
# ==========================================
st.title("Hurricane Hazard Analysis")

# Load Data
df = load_and_parse_data()
if df.empty:
    st.error(f"Failed to load data. Ensure '{LOCAL_FILENAME}' is in the same directory as this script.")
    st.stop()

# --- INPUT CONTROLS ---
st.markdown("### Configuration")
col1, col2, col3 = st.columns(3)
with col1:
    lat = st.number_input("Lat:", value=32.3078, format="%.4f")
    start_year = st.number_input("Start Year:", value=1851, step=1)
with col2:
    lon = st.number_input("Lon:", value=-64.7505, format="%.4f")
    min_wind = st.number_input("Min Wind (km/h):", value=63, step=1)
with col3:
    buffer_km = st.number_input("Buffer (km):", value=300, step=10)
    max_cap = st.number_input("Max Cap (km/h):", value=322, step=1)

mode = st.radio("Analysis Mode:", ["Empirical Fit", "Poisson + GEV"], horizontal=True)

# --- ANALYSIS ENGINE ---
analysis_years = 2024 - start_year + 1

# Filter Data
filtered_df = df[(df['Year'] >= start_year) & (df['WindSpeed_KMH'] >= min_wind)].copy()

if filtered_df.empty:
    st.warning("No data matches the current filters.")
    st.stop()

# Geospatial calculations
filtered_df['Dist'] = haversine_distance_vectorized(filtered_df['Lat'], filtered_df['Lon'], lat, lon)
inside_mask = filtered_df['Dist'] <= buffer_km

hit_ids = filtered_df[inside_mask]['StormID'].unique()
map_df = filtered_df[filtered_df['StormID'].isin(hit_ids)]
buffer_df = filtered_df[inside_mask]

def extract_max_winds(dataframe):
    grouped = dataframe.sort_values('WindSpeed_KMH', ascending=False).groupby('StormID').first().reset_index()
    return [{'Wind': r['WindSpeed_KMH'], 'Name': r['Name'], 'Year': int(r['Year'])} for _, r in grouped.iterrows()]

data_buf = extract_max_winds(buffer_df)
data_whole = extract_max_winds(map_df)

# Calculations
x_buf_emp, y_buf_emp = get_empirical_raw(data_buf, analysis_years)
x_whole_emp, y_whole_emp = get_empirical_raw(data_whole, analysis_years)

plot_rps = np.logspace(np.log10(1.1), np.log10(1000), 100)
y_buf_gev_plot = get_gev_curve(data_buf, analysis_years, plot_rps, max_cap)
y_whole_gev_plot = get_gev_curve(data_whole, analysis_years, plot_rps, max_cap)

sum_emp_buf = get_empirical_interpolated(data_buf, analysis_years, STANDARD_RPS)
sum_emp_whole = get_empirical_interpolated(data_whole, analysis_years, STANDARD_RPS)
sum_gev_buf = get_gev_curve(data_buf, analysis_years, STANDARD_RPS, max_cap)
sum_gev_whole = get_gev_curve(data_whole, analysis_years, STANDARD_RPS, max_cap)

# --- PLOTTING ---
fig = make_subplots(
    rows=2, cols=1, 
    row_heights=[0.6, 0.4], 
    vertical_spacing=0.05,  
    specs=[[{"type": "map"}], [{"type": "xy"}]],
    subplot_titles=(f"Tracks within {buffer_km}km ({len(hit_ids)} Events)", f"Hazard Curve: {mode}")
)

if mode == "Empirical Fit":
    fig.add_trace(go.Scatter(x=x_buf_emp, y=y_buf_emp, mode='lines+markers', line_shape='linear', name='Buffer Region', line=dict(color='blue', width=2), marker=dict(size=6)), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_whole_emp, y=y_whole_emp, mode='lines+markers', line_shape='linear', name='Whole Track', line=dict(color='red', width=2, dash='dot'), marker=dict(size=4)), row=2, col=1)
else:
    fig.add_trace(go.Scatter(x=x_buf_emp, y=y_buf_emp, mode='markers', name='Events (Buffer)', marker=dict(color='blue', size=6, opacity=0.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_whole_emp, y=y_whole_emp, mode='markers', name='Events (Whole)', marker=dict(color='red', size=4, opacity=0.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=plot_rps, y=y_buf_gev_plot, mode='lines', line_shape='spline', name='GEV Fit (Buffer)', line=dict(color='blue', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=plot_rps, y=y_whole_gev_plot, mode='lines', line_shape='spline', name='GEV Fit (Whole)', line=dict(color='red', width=2, dash='dot')), row=2, col=1)
    fig.add_trace(go.Scatter(x=[1.0, 1000], y=[max_cap, max_cap], mode='lines', line=dict(color='black', dash='dash', width=1), name='Max Cap', hoverinfo='skip'), row=2, col=1)

# Map Plotting
circ_lats, circ_lons = generate_circle_points(lat, lon, buffer_km)
fig.add_trace(go.Scattermap(lat=circ_lats, lon=circ_lons, mode='lines', line=dict(color='black', width=2), showlegend=False, hoverinfo='skip'), row=1, col=1)

lats, lons = [], []
for _, storm in map_df.groupby('StormID'):
    lats.extend(storm['Lat'].tolist() + [None])
    lons.extend(storm['Lon'].tolist() + [None])
fig.add_trace(go.Scattermap(lat=lats, lon=lons, mode='lines', line=dict(color='gray', width=1), opacity=0.4, showlegend=False, hoverinfo='skip'), row=1, col=1)

map_hover_text = buffer_df.apply(
    lambda r: f"<b>{r['Name']} ({r['Year']})</b><br>Cat: {r['Category']}<br>Wind: {r['WindSpeed_KMH']:.0f} km/h", axis=1
)

fig.add_trace(go.Scattermap(
    lat=buffer_df['Lat'], lon=buffer_df['Lon'], mode='markers',
    marker=dict(
        size=6, color=buffer_df['WindSpeed_KMH'], colorscale='YlOrRd', cmin=60, cmax=250, 
        showscale=True, colorbar=dict(title='km/h', len=0.4, y=0.8, x=1.02)
    ),
    text=map_hover_text, hoverinfo='text', name='Intensity'
), row=1, col=1)

fig.add_trace(go.Scattermap(lat=[lat], lon=[lon], mode='markers', marker=dict(size=10, color='black'), name='Target', hoverinfo='none'), row=1, col=1)

fig.update_layout(
    map=dict(style="carto-positron", center=dict(lat=lat, lon=lon), zoom=3),
    margin={"r":10,"t":30,"l":10,"b":10},
    legend=dict(orientation="h", y=1),
    height=800
)
fig.update_xaxes(title_text="Return Period (Years)", type="log", row=2, col=1)
fig.update_yaxes(title_text="Wind Speed (km/h)", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# --- EXCEL EXPORT ---
df_events = buffer_df[['StormID', 'Name', 'Year', 'Date', 'Time', 'Lat', 'Lon', 'WindSpeed_Knots', 'WindSpeed_KMH', 'Category']].to_dict('records')
curve_emp = [{'RP': x, 'Wind': y} for x, y in zip(x_buf_emp, y_buf_emp)]
curve_gev = [{'RP': x, 'Wind': y} for x, y in zip(plot_rps, y_buf_gev_plot)]

summary_data = []
for i, rp in enumerate(STANDARD_RPS):
    summary_data.append({
        'Return_Period': rp,
        'Empirical_Buffer': sum_emp_buf[i],
        'GEV_Buffer': sum_gev_buf[i],
        'Empirical_WholeTrack': sum_emp_whole[i],
        'GEV_WholeTrack': sum_gev_whole[i]
    })

excel_data = generate_excel_bytes(summary_data, df_events, curve_emp, curve_gev)

st.download_button(
    label="Download Excel Data",
    data=excel_data,
    file_name="hurricane_hazard_analysis.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    type="primary"
)