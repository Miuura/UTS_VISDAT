import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.stats import mannwhitneyu, ttest_ind
from plotly.subplots import make_subplots
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from scipy import stats

# Workaround: suppress a known RuntimeWarning that can appear in some
# Streamlit versions when cache expiration triggers an internal coroutine
# that isn't awaited. This is a targeted suppression for the exact
# message; the recommended long-term fix is to upgrade Streamlit to a
# version where the bug is resolved.
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="coroutine 'expire_cache' was never awaited"
)

# ======================================================================================
# Page Configuration
# ======================================================================================
st.set_page_config(
    page_title="Precision Farming Analytics Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================================================
# Data Loading
# ======================================================================================
@st.cache_data
def load_data():
    """Loads the agriculture dataset and performs initial preprocessing."""
    try:
        df = pd.read_csv('agriculture_dataset.csv')
        # Basic preprocessing
        df['Crop_Health_Label_Str'] = df['Crop_Health_Label'].map({1: 'Healthy', 0: 'Unhealthy'})
        df['Rainfall_Category'] = pd.cut(df['Rainfall'], bins=[0, 200, 600, np.inf], labels=['Low', 'Medium', 'High'])
        return df
    except FileNotFoundError:
        st.error("Dataset file 'agriculture_dataset.csv' not found. Please place it in the same directory as the script.")
        return None

df = load_data()

if df is None:
    st.stop()

# ======================================================================================
# Sidebar Navigation
# ======================================================================================
st.sidebar.title("🌾 Precision Farming Dashboard")
st.sidebar.markdown("An analytical tool for data-driven agricultural insights.")

page = st.sidebar.radio(
    "Select Analysis Case:",
    [
        "Dashboard Overview",
        "1. Crop Health Diagnostics",
        "2. Pest Outbreak Management",
        "3. Yield Optimization Factors",
        "4. Precision Irrigation Strategy",
        "5. Environmental Stress Analysis",
        "6. Disease Risk Assessment",
        "7. Resource Use Efficiency",
        "8. Growth Stage Vulnerability",
        "9. Soil Health Management",
        "10. Climate Impact on Crops"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("This dashboard is based on a comprehensive dataset of crop health and environmental factors.")

# ======================================================================================
# Main Panel
# ======================================================================================

# Overview Page
if page == "Dashboard Overview":
    st.title("🌾 Executive Summary: The State of Our Fields")
    st.markdown("""
    Welcome to your agricultural analytics dashboard. This page presents an executive summary of your farm's condition based on the latest data. Use these insights to understand the big picture and identify areas requiring further attention.
    
    **Select an analysis case from the sidebar for a deeper exploration.**
    """)

    st.markdown("---")

    # --- Key Performance Indicators (KPIs) ---
    st.header("Key Performance Indicators (KPIs)")
    col1, col2, col3, col4 = st.columns(4)
    
    total_data_points = len(df)
    healthy_pct = (df['Crop_Health_Label'].value_counts(normalize=True).get(1, 0) * 100)
    total_yield = df['Expected_Yield'].sum() / 1000 # to tonnes
    pest_hotspots_pct = (df[df['Pest_Hotspots'] > 0].shape[0] / total_data_points * 100)

    col1.metric("Total Observed Fields", f"{total_data_points:,}")
    col2.metric("Healthy Fields Percentage", f"{healthy_pct:.1f}%")
    col3.metric("Total Estimated Yield (tonnes)", f"{total_yield:,.0f}")
    col4.metric("Fields Affected by Pests", f"{pest_hotspots_pct:.1f}%", delta_color="inverse")

    st.markdown("---")

    # --- Detailed Analysis Section ---
    st.header("Detailed Analysis")
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Crop Type Distribution")
        crop_counts = df['Crop_Type'].value_counts().sort_values(ascending=True)
        fig = px.bar(crop_counts, 
                     x=crop_counts.values, 
                     y=crop_counts.index, 
                     orientation='h',
                     labels={'x': 'Number of Fields', 'y': 'Crop Type'},
                     text=crop_counts.values)
        fig.update_layout(showlegend=False, yaxis_title=None)
        fig.update_traces(textposition='inside', marker_color='#4C78A8')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("This chart shows the number of fields cultivated for each crop type. *Wheat* is the most cultivated crop.")

    with col2:
        st.subheader("Health Distribution by Crop")
        health_by_crop = df.groupby('Crop_Type')['Crop_Health_Label_Str'].value_counts(normalize=True).mul(100).rename('percentage').reset_index()
        fig = px.bar(health_by_crop, 
                     x='Crop_Type', 
                     y='percentage', 
                     color='Crop_Health_Label_Str',
                     barmode='group',
                     labels={'percentage': 'Percentage (%)', 'Crop_Type': 'Crop Type'},
                     color_discrete_map={'Unhealthy':'#E45756', 'Healthy':'#54A24B'})
        fig.update_layout(yaxis_title="Percentage of Fields (%)")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("A comparison of the percentage of healthy and unhealthy fields for each crop type. This helps identify which crops are most vulnerable.")

    st.markdown("<br>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Yield Potential by Crop")
        yield_by_crop = df.groupby('Crop_Type')['Expected_Yield'].mean().sort_values(ascending=False)
        fig = px.bar(yield_by_crop,
                     x=yield_by_crop.index,
                     y=yield_by_crop.values,
                     labels={'y': 'Average Yield (kg/ha)', 'x': 'Crop Type'},
                     color=yield_by_crop.values,
                     color_continuous_scale='Aggrnyl')
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("This chart displays the average potential yield for each crop type. *Maize* shows the highest potential yield per hectare.")

    with col4:
        st.subheader("Initial Insights & Recommendations")
        st.info(
            """
            **Key Findings:**
            1.  **Wheat Dominance:** Wheat is the most widely planted crop, making it a primary focus for optimization.
            2.  **Rice Vulnerability:** Rice shows a relatively high percentage of unhealthy fields compared to other crops, indicating potential systemic issues.
            3.  **Maize Productivity:** Although not as widely planted as wheat, maize has the highest potential yield per hectare, making it a valuable asset.

            **Initial Recommendations:**
            -   **Focus Investigation:** Use this dashboard to investigate why Rice has a lower health rate. Start with *Case 5 (Environmental Stress)* or *Case 6 (Disease Risk)*.
            -   **Improvement Opportunity:** Further analysis of Maize could reveal best practices that might be applicable to other crops to boost their yield.
            """
        )

# Case 1: Crop Health Diagnostics
elif page == "1. Crop Health Diagnostics":
    st.header("1. Crop Health Diagnostics")
    st.markdown("""
    **Problem:** Early and accurate detection of crop stress is vital to prevent yield loss. How can we use remote sensing data to distinguish healthy from unhealthy crops?
    """)

    # LEFT COLUMN – FILTERS
    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Filters")

        crop_options = ['All']
        if 'Crop_Type' in df.columns:
            crop_options += list(df['Crop_Type'].dropna().unique())

        selected_crop = st.selectbox("Select Crop Type", crop_options)

        st.subheader("Analysis Setup")
        st.markdown("""
        3D scatter using:  
        - **Soil_Moisture** (X)  
        - **Humidity** (Y)  
        - **Chlorophyll_Content** (Z)  
        Color = **Local Density (k-NN)** to view plant health clusters.
        """)

        sample_size = st.slider("Sample size", 500, 20000, 5000, step=500)
        n_neighbors = st.slider("k for Density (k-NN)", 5, 100, 20)
        opacity = st.slider("Opacity", 0.1, 1.0, 0.8)

    # RIGHT COLUMN – 3D SCATTER
    with col2:
        # 5 feature dataset
        needed = [
            "Crop_Health_Label",
            "Soil_Moisture",
            "Humidity",
            "Chlorophyll_Content",
            "Temperature"
        ]

        missing = [c for c in needed if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            st.stop()

        plot_df = df.copy()
        if selected_crop != 'All':
            plot_df = plot_df[plot_df['Crop_Type'] == selected_crop]

        n_take = min(sample_size, len(plot_df))
        plot_sample = plot_df.sample(n=n_take, random_state=42) if len(plot_df) > n_take else plot_df.copy()

        imputer = SimpleImputer(strategy="median")
        plot_sample[["Soil_Moisture", "Humidity", "Chlorophyll_Content"]] = imputer.fit_transform(
            plot_sample[["Soil_Moisture", "Humidity", "Chlorophyll_Content"]]
        )

        scaler = StandardScaler()
        X = scaler.fit_transform(
            plot_sample[["Soil_Moisture", "Humidity", "Chlorophyll_Content"]].astype(float)
        )

        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
        dist, _ = nbrs.kneighbors(X)

        if np.allclose(dist[:, 0], 0.0):
            dist = dist[:, 1:]


        density = 1.0 / (dist.mean(axis=1) + 1e-12)
        density = (density - density.min()) / (density.max() - density.min() + 1e-12)

        plot_sample = plot_sample.reset_index(drop=True)
        plot_sample["density"] = density.astype(float)

        plot_sample["_label_str"] = plot_sample["Crop_Health_Label"].map({
            1: "Healthy",
            0: "Unhealthy"
        }).astype(str)

        for c in ["Soil_Moisture", "Humidity", "Chlorophyll_Content", "density"]:
            plot_sample[c] = plot_sample[c].astype(float)

        st.subheader("3D Scatter Plot with Density")

        fig3d = px.scatter_3d(
            plot_sample,
            x="Soil_Moisture",
            y="Humidity",
            z="Chlorophyll_Content",
            color="density",
            color_continuous_scale="Turbo",
            opacity=opacity,
            symbol="_label_str",
            hover_data={
                "Soil_Moisture": True,
                "Humidity": True,
                "Chlorophyll_Content": True,
                "Crop_Health_Label": True,
                "density": ':.4f'
            }
        )

        fig3d.update_traces(
            marker=dict(size=4, line=dict(width=0.5, color="black"))
        )

        fig3d.update_layout(
            height=600,
            width=1000,
            margin=dict(l=0, r=0, t=40, b=0),
            coloraxis_colorbar=dict(title="Density")
        )

        st.plotly_chart(fig3d, use_container_width=False)

    # STATISTICAL VALIDATION (T-TEST)
    st.subheader("Statistical Validation (Independent t-test)")

    t_df = df.dropna(subset=[
        "Soil_Moisture",
        "Humidity",
        "Chlorophyll_Content",
        "Crop_Health_Label"
    ]).copy()

    healthy = t_df[t_df["Crop_Health_Label"] == 1]

    unhealthy = t_df[t_df["Crop_Health_Label"] == 0]

    def ttest(col):
        a = healthy[col]; b = unhealthy[col]
        t, p = stats.ttest_ind(a, b, equal_var=False)
        d = (a.mean() - b.mean()) / np.sqrt(((a.std()**2)+(b.std()**2))/2)
        return t, p, a.mean(), b.mean(), d

    rows = []
    for col in ["Soil_Moisture", "Humidity", "Chlorophyll_Content"]:
        t, p, m1, m0, d = ttest(col)
        rows.append([col, t, p, m1, m0, d])

    ttable = pd.DataFrame(rows, columns=[
        "Feature", "t", "p-value", "Mean Healthy", "Mean Unhealthy", "Effect Size (d)"
    ])

    st.dataframe(ttable.round(4), use_container_width=True)

    # FEATURE IMPORTANCE (Fixed Values)
    st.subheader("Feature Importance (Provided)")

    fi = pd.DataFrame({
        "Feature": ["Crop_Health_Label", "Soil_Moisture", "Humidity", "Chlorophyll_Content"],
        "Importance": [0.639968, 0.084419, 0.040782, 0.040276]
    })

    st.dataframe(fi, use_container_width=True)

    # INSIGHTS & RECOMMENDATIONS
    st.subheader("Professional Insights & Recommendations")

    c1, c2 = st.columns(2)

    with c1:
        st.info("""
        **Key Insight:**  
        - The density-based 3D visualization reveals that healthy crops consistently cluster around stable environmental conditions — balanced soil moisture, moderate humidity levels, and high chlorophyll content. 
        - Unhealthy crops form dense pockets when any of these parameters deviate sharply. 
        - These clusters also highlight potential early-warning zones where crop stress begins before symptoms appear visually.
        """)
        st.success("""
        **Benefit:**  
        - Enables early detection of stress hotspots before yield loss occurs.  
        - Allows farmers or agronomists to allocate interventions more efficiently (targeted irrigation, pest control, nutrient adjustments).  
        - Minimizes operational costs by focusing effort only on the highest-risk regions.  
        - Improves decision-making with clearer segmentation of healthy vs. unhealthy crop groups.  
        """)

    with c2:
        st.warning("""
        **AI/ML Potential:**  
        - The chosen features (Soil Moisture, Humidity, Chlorophyll Content, Temperature) provide a strong foundation for building machine learning classifiers (Random Forest, XGBoost, LightGBM).  
        - Density values can be used as an additional engineered feature, improving signal strength for early stress detection.  
        - Integration with time-series sensing enables models such as LSTM or Temporal CNN to learn patterns of gradual stress buildup.  
        """)
        st.error("""
        **Risk:**  
        - **Sensor Noise:** Incorrect readings can distort density, leading to false stress detection.   
        - **Overfitting to Environmental Conditions:** Models may latch onto season-specific behavior.   
        - **Imbalanced Data:** If healthy crops dominate the dataset, unhealthy clusters may be under-represented.  
        """)

# Case 2: Pest Outbreak Management
elif page == "2. Pest Outbreak Management":
    st.header("2. Pest Outbreak Management")
    st.markdown("""
    **Problem:** Pest damage is a primary cause of yield loss. How can we quantify the economic impact of pest damage and identify which crops are most vulnerable?
    """)

    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Filters")
        crop_list = ['All']
        if 'Crop_Type' in df.columns:
            crop_list += list(df['Crop_Type'].dropna().unique())
        selected_crop_pest = st.selectbox("Select Crop Type", crop_list, key="pest_crop")

        st.subheader("Analysis")
        st.markdown("""
        This scatter plot directly correlates **Pest Damage (%)** with **Expected Yield (kg/ha)**.
        It overlays a fitted linear trend (predicted yield) and calculates estimated economic loss using a market price per kg.
        Each point represents a field observation; color = crop type.
        """)

        sample_size = st.slider("Sample size for plotting (max points to consider)", 200, 20000, 5000, step=200)
        market_price = st.number_input("Market price (currency per kg) — used to compute economic loss", min_value=0.0, value=0.25, step=0.01, format="%.4f")
        show_trend = st.checkbox("Show linear trend line (overall)", value=True)
        show_crop_slopes = st.checkbox("Show per-crop vulnerability table (slope)", value=True)

    with col2:
        plot_df = df.copy()
        required_cols = ['Pest_Damage', 'Expected_Yield', 'Crop_Type']
        missing = [c for c in required_cols if c not in plot_df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}. Please ensure dataset contains Pest_Damage, Expected_Yield, and Crop_Type.")
        else:
            if selected_crop_pest != 'All':
                plot_df = plot_df[plot_df['Crop_Type'] == selected_crop_pest]

            n_take = min(sample_size, len(plot_df))
            plot_sample = plot_df.sample(n=n_take, random_state=42) if len(plot_df) > n_take else plot_df.copy()

            imputer = SimpleImputer(strategy="median")
            plot_sample[['Pest_Damage', 'Expected_Yield']] = imputer.fit_transform(plot_sample[['Pest_Damage', 'Expected_Yield']])

            plot_sample['Pest_Damage'] = plot_sample['Pest_Damage'].astype(float)
            plot_sample['Expected_Yield'] = plot_sample['Expected_Yield'].astype(float)

            fig_scatter = px.scatter(
                plot_sample,
                x='Pest_Damage',
                y='Expected_Yield',
                color='Crop_Type',
                title=f'Impact of Pest Damage on Yield for {selected_crop_pest} Crops',
                labels={'Pest_Damage': 'Pest Damage (%)', 'Expected_Yield': 'Expected Yield (kg/ha)'},
                hover_data=['Crop_Type']
            )

            fig = make_subplots(specs=[[{"secondary_y": True}]])

            for trace in fig_scatter.data:
                fig.add_trace(trace, secondary_y=False)

            if show_trend and len(plot_sample) >= 2:
                x = plot_sample['Pest_Damage'].to_numpy()
                y = plot_sample['Expected_Yield'].to_numpy()
                coeffs = np.polyfit(x, y, 1)  
                slope, intercept = coeffs[0], coeffs[1]

                x_line = np.linspace(plot_sample['Pest_Damage'].min(), plot_sample['Pest_Damage'].max(), 200)
                y_line = slope * x_line + intercept

                fig.add_trace(
                    go.Scatter(x=x_line, y=y_line, mode='lines', name='Linear trend (yield)', line=dict(color='black', width=2, dash='dash')),
                    secondary_y=False
                )

                baseline_yield = intercept  
                econ_loss = (baseline_yield - y_line) * float(market_price)
                econ_loss = np.maximum(econ_loss, 0.0)

                fig.add_trace(
                    go.Scatter(x=x_line, y=econ_loss, mode='lines', name='Estimated economic loss', line=dict(color='firebrick', width=2)),
                    secondary_y=True
                )

                slope_per_10 = slope * 10
                st.markdown(f"**Overall trend:** slope = {slope:.3f} kg/ha per 1% pest damage (≈ {slope_per_10:.1f} kg/ha per 10%). Baseline yield (0% pest) ≈ {baseline_yield:.1f} kg/ha.")

            fig.update_xaxes(title_text="Pest Damage (%)")
            fig.update_yaxes(title_text="Expected Yield (kg/ha)", secondary_y=False)
            fig.update_yaxes(title_text=f"Estimated Economic Loss (currency per ha) — price={market_price:.4f} per kg", secondary_y=True)

            fig.update_layout(height=600, width=900, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01))

            st.plotly_chart(fig, use_container_width=True)

    if show_crop_slopes:
        st.subheader("Crop vulnerability (linear slope per crop)")
        slopes = []
        for crop, g in df.dropna(subset=['Pest_Damage', 'Expected_Yield']).groupby('Crop_Type'):
            if len(g) >= 5:
                xg = g['Pest_Damage'].astype(float).to_numpy()
                yg = g['Expected_Yield'].astype(float).to_numpy()
                m, b = np.polyfit(xg, yg, 1)
                slopes.append({'Crop_Type': crop, 'slope': float(m), 'count': len(g)})
        if len(slopes) == 0:
            st.info("Not enough data per crop to compute slopes. Need >=5 observations per crop.")
        else:
            slopes_df = pd.DataFrame(slopes)
            slopes_df['abs_slope'] = slopes_df['slope'].abs()
            slopes_df = slopes_df.sort_values('slope')
            st.dataframe(slopes_df[['Crop_Type', 'slope', 'count']].rename(columns={'slope': 'slope (kg/ha per % damage)'}).round(4), use_container_width=True)
            top = slopes_df.head(1).iloc[0]
            st.markdown(f"**Most vulnerable crop:** {top['Crop_Type']} (slope = {top['slope']:.3f} kg/ha per 1% pest damage)")

    # --- Feature Importance ---
    st.subheader("Feature Importance Insights")

    st.markdown("**Expected Yield (Regression) — Feature Importance**")
    yield_features = {
        'NDVI': 0.174586,
        'Chlorophyll_Content': 0.174343,
        'Temperature': 0.174252,
        'Soil_Moisture': 0.173751,
        'Humidity': 0.173177,
        'Pest_Damage': 0.129890
    }
    yield_df = pd.DataFrame({
        'Feature': list(yield_features.keys()),
        'Importance': list(yield_features.values())
    }).sort_values('Importance', ascending=False)
    st.dataframe(yield_df.style.format({"Importance": "{:.4f}"}), use_container_width=True)

    st.markdown("**Pest Vulnerability (Classification) — Feature Importance**")
    pest_features = {
        'Humidity': 0.200288,
        'Temperature': 0.200040,
        'Soil_Moisture': 0.200002,
        'NDVI': 0.199974,
        'Chlorophyll_Content': 0.199696
    }
    pest_df = pd.DataFrame({
        'Feature': list(pest_features.keys()),
        'Importance': list(pest_features.values())
    }).sort_values('Importance', ascending=False)
    st.dataframe(pest_df.style.format({"Importance": "{:.6f}"}), use_container_width=True)

    # --- Professional Insights & Recommendations ---
    st.subheader("Professional Insights & Recommendations")
    col1a, col2a = st.columns(2)
    with col1a:
        st.info("""
        **Key Insight:**
        The trendline shows a clear negative relationship: higher pest damage leads to lower expected yield. By converting yield drops into monetary loss using market price, we quantify economic impact per unit damage.
        """)
        st.success("""
        **Potential Benefit:**
        Using this visualization, decision-makers can prioritize interventions on crops/fields where the slope (vulnerability) is steepest — those provide the highest ROI when treated early.
        """)
    with col2a:
        st.warning("""
        **AI & ML Potential:**
        1. Train regression models to predict yield loss from pest damage and environmental features; use predictions to estimate economic loss at scale.
        2. Combine with spatial data (field polygons) to create loss heatmaps for resource allocation.
        """)
        st.error("""
        **Potential Risk:**
        Over-simplifying with a single linear trend ignores non-linear thresholds and interactions (e.g., stage-of-growth). Ground-truth validation and season-aware models are essential.
        """)
        
# Case 3: Yield Optimization Factors
elif page == "3. Yield Optimization Factors":
    st.header("3. Yield Optimization Factors")
    st.markdown("""
    **Problem:** Maximizing yield is a complex puzzle with many interacting variables. Which environmental and soil factors have the strongest influence on yield?
    """)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Filters")
        num_features = st.slider(
            "Number of top features to display",
            min_value=3,
            max_value=16,
            value=16
        )

    with col2:
        st.subheader("Analysis")
        st.markdown("""
        This bar chart shows the features with the strongest correlation to **Expected Yield**. Bars pointing to the right indicate a positive correlation (increasing the feature tends to increase yield), while bars pointing left indicate a negative correlation. This provides a clear, ranked view of the most impactful factors.
        """)
        
    corr_matrix = df.select_dtypes(include=np.number).corr()
    yield_corr = corr_matrix[['Expected_Yield']].drop('Expected_Yield').sort_values(by='Expected_Yield', ascending=False)
    
    top_pos = yield_corr.head(num_features)
    top_neg = yield_corr.tail(num_features)
    
    # Combine for plotting
    plot_corr = pd.concat([top_pos, top_neg]).sort_values(by='Expected_Yield')

    fig = px.bar(plot_corr,
                    x='Expected_Yield',
                    y=plot_corr.index,
                    orientation='h',
                    title=f'Top {num_features*2} Factors Correlated with Expected Yield',
                    color='Expected_Yield',
                    color_continuous_scale='RdBu_r',
                    labels={'Expected_Yield': 'Correlation Coefficient', 'y': 'Feature'},
                    height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Professional Insights & Recommendations")
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **Key Insight:**
        `NDVI` and `Chlorophyll_Content` are the strongest positive predictors of yield, confirming their role as primary health indicators. Conversely, `Crop_Stress_Indicator` and `Pest_Damage` are the strongest negative predictors. This highlights the dual strategy for yield optimization: maximizing vegetative health while minimizing stress and damage.
        """)
        st.success("""
        **Potential Benefit:**
        Focusing resources on the top 3 correlated factors (e.g., optimizing conditions for high NDVI) can be a highly efficient strategy, potentially boosting overall yield by **10-15%** by prioritizing actions with the highest expected impact.
        """)
    with col2:
        st.warning("""
        **AI & ML Potential:**
        A **Regression Model** (e.g., Gradient Boosting or a simple Neural Network) can be built to predict `Expected_Yield`. Feature importance analysis from this model (like SHAP values) would provide deeper, non-linear insights into the key drivers of yield, going beyond simple linear correlation.
        """)
        st.error("""
        **Potential Risk:**
        Correlation does not imply causation. For example, high temperature might negatively correlate with yield, but the true cause could be the associated decrease in soil moisture. A simple correlation view can be misleading if not interpreted with domain knowledge.
        """)

# Case 4: Precision Irrigation Strategy
elif page == "4. Precision Irrigation Strategy":
    st.header("4. Precision Irrigation Strategy")
    st.markdown("""
    **Problem:** Over- or under-watering are common issues that stress plants and waste water. What is the optimal soil moisture range for different crops?
    """)

    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Filters")
        selected_crop_irrigation = st.selectbox("Select Crop Type", df['Crop_Type'].unique(), key="irrigation_crop")
        
    with col2:
        st.subheader("Analysis")
        st.markdown("""
        This scatter plot visualizes the relationship between **Expected Yield** and **NDVI**, with points colored by 5 bins of **Soil Moisture**. This multi-variate analysis aims to identify the specific moisture range that co-occurs with high yield and high vegetative health.
    """)
        
    plot_df = df[df['Crop_Type'] == selected_crop_irrigation]
    
    plot_df['moisture_cat'] = pd.cut(df['Soil_Moisture'], bins=5)
    
    fig = px.scatter(
        plot_df,
        x='Expected_Yield',
        y='NDVI',
        color='moisture_cat',
        color_continuous_scale=px.colors.sequential.Plasma,
        opacity=0.3,
        title=f'Soil Moisture Distribution for Healthy vs. Unhealthy {selected_crop_irrigation}',
        height=600
    )
        
    fig.update_layout(
        font_family="sans-serif",
        title_font_size=20,
        legend_title_text='Soil Moisture Bin'
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgray')
    st.plotly_chart(fig, use_container_width=True)

    # Statistical Analysis
    from scipy.stats import f_oneway # Ensure this import is present

    st.subheader("Statistical Validation (ANOVA on Expected Yield by Moisture Bin)")

    bin_data = []
    unique_bins = sorted(plot_df['moisture_cat'].dropna().unique())

    for bin_label in unique_bins:
        yields = plot_df[plot_df['moisture_cat'] == bin_label]['Expected_Yield'].dropna()
        if len(yields) > 1:
            bin_data.append(yields)

    st.markdown(f"**Average Expected Yield per Soil Moisture Bin for {selected_crop_irrigation}:**")

    yield_means = {}
    for i, bin_label in enumerate(unique_bins):
        if i < len(bin_data):
            mean_yield = bin_data[i].mean()
            yield_means[bin_label] = mean_yield
            st.markdown(f"- **Bin {bin_label}:** `{mean_yield:.2f}`")

    # ANOVA
    if len(bin_data) >= 2 and all(len(b) > 1 for b in bin_data):
        stat, p_value = f_oneway(*bin_data)

        st.markdown(f"""
        - **P-value (ANOVA):** `{p_value:.2e}`
        """)

        if p_value < 0.05:
            optimal_bin = max(yield_means, key=yield_means.get)
            st.success(f"The difference in average yield among the 5 soil moisture categories is **statistically significant** (P < 0.05). The bin with the highest average yield is **{optimal_bin}**.")
        else:
            st.warning("The difference in yield across soil moisture categories is not statistically significant, suggesting moisture is not the primary limiting factor for this crop.")
    else:
        st.warning("Insufficient data to perform ANOVA on all soil moisture categories.")
        
    st.subheader("Professional Insights & Recommendations")
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **Key Insight:**
        The visualization reveals a strong positive correlation between **NDVI** (vegetative health) and **Expected Yield**. The highest density of data points (the 'sweet spot') occurs where the color representing the **intermediate/optimal** soil moisture bins is concentrated. This confirms that an **optimal moisture range** exists, but its boundaries are narrow and critical for maximizing both health and output.
        """)
        st.success("""
        **Potential Benefit (Precision Irrigation):**
        Identifying the **highest-yielding moisture bin** (e.g., (17.0, 23.0)) provides **specific numerical limits** for irrigation control. By programming smart irrigation systems to maintain soil moisture strictly within this optimal range, farmers can mitigate water stress (drought/saturation) and ensure water resources directly contribute to maximizing yield potential.
        """)
    with col2:
        st.warning("""
        **AI & ML Potential:**
        Since the plot shows significant overlap across all 5 moisture bins in the high-yield area, it suggests that **moisture is a supporting factor, not a single determinant.** A robust **Yield Prediction Regression Model** should be trained using a combination of inputs like **Soil Moisture, Temperature, and Soil pH** simultaneously to achieve the highest accuracy in advising irrigation needs.
        """)
        st.error("""
        **Potential Risk:**
        While an optimal bin is identified, this range can shift dynamically depending on the **crop's growth stage** (`Crop_Growth_Stage`). Using static bin limits throughout the entire season could lead to over-watering in early stages or under-watering during peak vegetative phases. The irrigation strategy must be **adaptive** and incorporate temporal growth data.
        """)

# Case 5: Environmental Stress Analysis
elif page == "5. Environmental Stress Analysis":
    st.header("5. Environmental Stress Analysis")
    st.markdown("""
    **Problem:** Crop stress is often a result of multiple interacting environmental factors. How do the distributions of these factors differ between low- and high-stress crops?
    """)

    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Filters")
        stress_threshold = st.slider("Select Stress Indicator Threshold", 0, 100, 70)
        st.markdown(f"**Current Threshold:** `{stress_threshold}`")
        
        st.subheader("Analysis")
        st.markdown("""
        These box plots compare the distribution of key environmental factors for crops above and below the selected stress threshold. This allows for a direct comparison of not just the median, but also the range and variability of conditions that lead to stress.
        """)

    with col2:
        stress_df = df.copy()
        stress_df['Stress_Level'] = stress_df['Crop_Stress_Indicator'].apply(lambda x: 'High Stress' if x > stress_threshold else 'Low Stress')
        
        env_factors = ['Temperature', 'Humidity', 'Rainfall', 'Wind_Speed']
        factor_units = {'Temperature': '°C', 'Humidity': '%', 'Rainfall': 'mm', 'Wind_Speed': 'km/h'}
        
        fig = make_subplots(rows=2, cols=2, subplot_titles=env_factors)
        
        for i, factor in enumerate(env_factors):
            row, col = (i // 2) + 1, (i % 2) + 1
            filtered_stress_df = stress_df[stress_df['Stress_Level'].isin(['High Stress', 'Low Stress'])]
            sub_fig = px.box(filtered_stress_df, x='Stress_Level', y=factor, color='Stress_Level', 
                             color_discrete_map={'High Stress': '#E45756', 'Low Stress': '#4C78A8'})
            for trace in sub_fig.data:
                fig.add_trace(trace, row=row, col=col)
            
            fig.update_yaxes(title_text=f"{factor} ({factor_units.get(factor, '')})", row=row, col=col)

        fig.update_layout(
            height=600, 
            showlegend=False, 
            title_text="Environmental Factor Distributions for Low vs. High Stress Crops",
            transition_duration=500 # Animation smoothing
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Professional Insights & Recommendations")
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **Key Insight:**
        High-stress situations are clearly associated with higher median temperatures, higher wind speeds, and lower rainfall. Interestingly, the humidity range for high-stress crops is tighter and lower than for low-stress crops, suggesting that dry air, when combined with other factors, is a major stress contributor.
        """)
        st.success("""
        **Potential Benefit:**
        By understanding the specific "stress signature" for a region, farmers can implement targeted mitigation strategies. For example, if high wind and high temperature are the main drivers, investing in windbreaks and shade nets could be more effective than simply increasing irrigation.
        """)
    with col2:
        st.warning("""
        **AI & ML Potential:**
        A **Clustering Algorithm** (e.g., K-Means or DBSCAN) could be used to automatically identify different "types" of environmental stress from the data (e.g., "Drought & Heat Stress", "Wind & Waterlog Stress"). This allows for more nuanced management strategies than a simple high/low classification.
        """)
        st.error("""
        **Potential Risk:**
        This analysis averages conditions, which might mask the impact of short, extreme events (e.g., a sudden frost or a heatwave lasting a few hours). Time-series analysis is needed to complement this view and capture the dynamics of stress induction.
        """)

# Case 6: Disease Risk Assessment
elif page == "6. Disease Risk Assessment":
    st.header("6. Disease Risk Assessment")
    st.markdown("""
    **Problem:** Many fungal and bacterial diseases thrive under specific temperature and humidity conditions. Can we identify these high-risk "danger zones"?
    """)

    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Filters")
        selected_crop_disease = st.selectbox("Select Crop Type", ['All'] + list(df['Crop_Type'].unique()), key="disease_crop")
        
        st.subheader("Analysis")
        st.markdown("""
        This 2D density heatmap shows the average health status of crops at different combinations of **Temperature** and **Humidity**. The highlighted red area represents the "danger zone" where disease risk is highest.
        """)

    with col2:
        plot_df = df.copy()
        if selected_crop_disease != 'All':
            plot_df = df[df['Crop_Type'] == selected_crop_disease]

        fig = px.density_heatmap(
            plot_df,
            x="Temperature",
            y="Humidity",
            z="Crop_Health_Label",
            histfunc="avg",
            color_continuous_scale="RdYlGn",
            title=f'Disease Risk Zone: Avg. Health by Temp & Humidity for {selected_crop_disease}',
            labels={
                "Temperature": "Temperature (°C)",
                "Humidity": "Humidity (%)"
            },
            color_continuous_midpoint=0.5,
            nbinsx=20, nbinsy=20
        )
        
        fig.update_coloraxes(colorbar_title="Health Index<br>(0=Poor, 1=Healthy)")

        fig.add_shape(
            type="rect",
            x0=30, y0=75, x1=plot_df['Temperature'].max(), y1=plot_df['Humidity'].max(),
            line=dict(color="Black", width=2, dash="dash"),
            fillcolor="rgba(255, 0, 0, 0.15)"
        )
        fig.add_annotation(
            x=35, y=85,
            text="High Disease Risk Zone",
            showarrow=False,
            font=dict(color="black", size=12, weight="bold"),
            bgcolor="rgba(255, 255, 255, 0.7)"
        )
        
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Professional Insights & Recommendations")
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **Key Insight:**
        A clear "danger zone" is visible at high humidity (>75%) combined with moderate-to-high temperatures (20-30°C). This is a classic incubator for fungal pathogens. Conversely, conditions of low humidity act as a natural protectant, even at high temperatures.
        """)
        st.success("""
        **Potential Benefit:**
        This insight allows for the creation of a highly targeted, preventative spraying schedule. Fungicides need only be applied when weather forecasts predict entry into this danger zone, potentially reducing chemical use by **over 50%** and preventing outbreaks before they start.
        """)
    with col2:
        st.warning("""
        **AI & ML Potential:**
        An **Anomaly Detection** model can be trained on the time-series data of healthy fields. When a field's temperature/humidity trajectory deviates and enters the "danger zone" for a sustained period, the model can raise an alert for a high-risk disease event.
        """)
        st.error("""
        **Potential Risk:**
        This analysis does not account for airflow (wind speed), which can significantly mitigate the effects of high humidity. A more advanced model should include wind as a third variable. Additionally, microclimates within a field can vary, so sensor placement is key.
        """)

# Case 7: Resource Use Efficiency
elif page == "7. Resource Use Efficiency":
    st.header("7. Resource Use Efficiency")
    st.markdown("""
    **Problem:** Are we getting the most "bang for our buck" from soil inputs? How does soil organic matter and pH level interact to affect yield?
    """)

    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Filters")
        selected_crop_resource = st.selectbox("Select Crop Type", df['Crop_Type'].unique(), key="resource_crop")
        zone_mode = st.radio("Optimal Zone Mode", ["Guideline (Agronomy)", "Data-driven (Top Yield)"], index=0, key="resource_zone_mode")
        if zone_mode == "Data-driven (Top Yield)":
            ph_bins = st.slider("Number of Soil pH bins", 8, 30, 15, key="resource_ph_bins")
            om_bins = st.slider("Number of Organic Matter bins", 8, 30, 15, key="resource_om_bins")
            top_quantile = st.slider("Top yield quantile for optimal zone", 60, 95, 80, step=5, key="resource_top_q")
        
        st.subheader("Analysis")
        st.markdown("""
        This section visualizes the relationship between Soil pH, Organic Matter, and Expected Yield.
        - In **Guideline (Agronomy)** mode, the optimal zone is a standard agronomic range (pH ~6.0–7.0, OM ~2.5–5.0%).
        - In **Data-driven (Top Yield)** mode, the optimal zone is computed from your data as the top yield quantile over a pH–OM grid.
        """)

    with col2:
        plot_df = df[df['Crop_Type'] == selected_crop_resource]
        sample_df = plot_df.sample(min(2000, len(plot_df)))

        if zone_mode == "Guideline (Agronomy)":
            fig = px.scatter_3d(
                sample_df,
                x='Soil_pH',
                y='Organic_Matter',
                z='Expected_Yield',
                color='Expected_Yield',
                color_continuous_scale='Viridis',
                title=f'Yield vs. Soil pH & Organic Matter for {selected_crop_resource}',
                labels={
                    "Soil_pH": "Soil pH",
                    "Organic_Matter": "Organic Matter (%)",
                    "Expected_Yield": "Expected Yield (kg/ha)"
                }
            )
            # Add a 3D shape to represent the guideline optimal zone
            fig.add_trace(go.Mesh3d(
                x=[6.0, 7.0, 7.0, 6.0, 6.0, 7.0, 7.0, 6.0],
                y=[2.5, 2.5, 5.0, 5.0, 2.5, 2.5, 5.0, 5.0],
                z=[sample_df['Expected_Yield'].min()] * 4 + [sample_df['Expected_Yield'].max()] * 4,
                i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                opacity=0.1,
                color='cyan',
                name='Optimal Zone (Guideline)',
                showlegend=True
            ))
            fig.update_layout(
                scene=dict(
                    xaxis_title='Soil pH',
                    yaxis_title='Organic Matter (%)',
                    zaxis_title='Expected Yield (kg/ha)'
                ),
                legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.1)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Data-driven: grid pH–OM, compute median yield per cell, highlight top quantile cells
            grid_df = plot_df[['Soil_pH', 'Organic_Matter', 'Expected_Yield']].dropna().copy()
            if len(grid_df) == 0:
                st.warning("No data available to compute data-driven optimal zone.")
            else:
                grid_df['ph_bin'] = pd.cut(grid_df['Soil_pH'], bins=ph_bins)
                grid_df['om_bin'] = pd.cut(grid_df['Organic_Matter'], bins=om_bins)
                agg = grid_df.groupby(['ph_bin', 'om_bin'])['Expected_Yield'].median().reset_index(name='median_yield')
                # Bin centers for plotting
                def bin_center(interval):
                    try:
                        return (interval.left + interval.right) / 2
                    except Exception:
                        return np.nan
                agg['ph_center'] = agg['ph_bin'].apply(bin_center)
                agg['om_center'] = agg['om_bin'].apply(bin_center)
                agg = agg.dropna(subset=['ph_center', 'om_center'])
                if len(agg) == 0:
                    st.warning("Insufficient binned data to visualize.")
                else:
                    # Determine threshold for top quantile
                    q = np.nanpercentile(agg['median_yield'], top_quantile)
                    agg['is_optimal'] = agg['median_yield'] >= q
                    # Create base heatmap of median yield
                    heatmap_fig = go.Figure()
                    heatmap_fig.add_trace(go.Heatmap(
                        x=agg['ph_center'],
                        y=agg['om_center'],
                        z=agg['median_yield'],
                        colorscale='Viridis',
                        colorbar_title='Median Yield'
                    ))
                    # Overlay mask for optimal cells (top quantile)
                    # Use a second heatmap with transparent color for 0 and cyan for 1
                    mask_z = agg['is_optimal'].astype(int)
                    overlay_colorscale = [
                        [0.0, 'rgba(0,0,0,0)'],
                        [0.5, 'rgba(0,0,0,0)'],
                        [1.0, 'rgba(0,255,255,0.35)']
                    ]
                    heatmap_fig.add_trace(go.Heatmap(
                        x=agg['ph_center'],
                        y=agg['om_center'],
                        z=mask_z,
                        colorscale=overlay_colorscale,
                        showscale=False
                    ))
                    heatmap_fig.update_layout(
                        title=f'Data-driven Optimal Zone for {selected_crop_resource} (Top {top_quantile}% median yield)',
                        xaxis_title='Soil pH (bin center)',
                        yaxis_title='Organic Matter (%) (bin center)',
                    )
                    # Add explanatory annotation
                    heatmap_fig.add_annotation(
                        xref="paper", yref="paper", x=0.01, y=1.08, showarrow=False,
                        text="Cyan overlay = cells in top yield quantile (data-driven optimal zone)"
                    )
                    st.plotly_chart(heatmap_fig, use_container_width=True)

    st.subheader("Professional Insights & Recommendations")
    col1, col2 = st.columns(2)
    with col1:
        if zone_mode == "Data-driven (Top Yield)":
            st.info("""
            **Key Insight (data-driven):**
            The cyan overlay marks pH–OM cells whose median yield lies in the top quantile for the selected crop. Near‑neutral pH (~6–7) combined with OM around ≥3% most often forms the optimal band; extreme pH rarely enters the top-yield zone even with high OM.
            """)
            st.success("""
            **Potential Benefit:**
            Steer fields toward the cyan band: correct pH toward 6–7 first, then build OM to ≥3%+. This sequencing maximizes the chance of reaching top‑yield conditions and avoids wasting inputs when pH is off.
            """)
        else:
            st.info("""
            **Key Insight (guideline):**
            Yield generally peaks near‑neutral pH (6.5–7.0) with OM in the 3–5% range. Adding OM alone does not compensate for improper pH—both must be managed together.
            """)
            st.success("""
            **Potential Benefit:**
            Use the guideline zone to prioritize **pH correction** on acidic soils, then target OM in deficient patches. This improves input efficiency and can reduce amendment costs by **15–20%**.
            """)
    with col2:
        st.warning("""
        **AI & ML Potential:**
        Fit a **surface model** (e.g., Gaussian Process/Gradient Boosting) on yield ~ pH + OM to derive optimized contours and quantify uncertainty. Use SHAP/partial dependence to explain why specific pH–OM ranges are optimal for each crop.
        """)
        st.error("""
        **Potential Risk:**
        Results depend on binning/sampling density (data-driven mode) and ignore soil texture/salinity/drainage. Validate with field context, ensure minimum data per cell, and avoid extrapolating beyond observed pH–OM ranges.
        """)

# Case 8: Growth Stage Vulnerability
elif page == "8. Growth Stage Vulnerability":
    st.header("8. Growth Stage Vulnerability")
    st.markdown("""
    **Problem:** A plant's vulnerability to stress changes as it matures. Which growth stages are the most critical to monitor and protect, and what is the typical stress profile at each stage?
    """)

    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Filters")
        selected_crop_growth = st.selectbox("Select Crop Type", df['Crop_Type'].unique(), key="growth_crop")
        
        st.subheader("Analysis")
        st.markdown("""
        This view shows the **median stress by growth stage** with error bars for variability (IQR). Labels display the share of fields above a high-stress threshold, clarifying which stage is most vulnerable.
        """)

    with col2:
        plot_df = df[df['Crop_Type'] == selected_crop_growth].copy()
        
        if not plot_df.empty:
            stage_map = {
                1: '1. Germination',
                2: '2. Vegetative',
                3: '3. Flowering',
                4: '4. Grain Filling/Fruiting'
            }
            
            plot_df['Growth_Stage_Desc'] = plot_df['Crop_Growth_Stage'].map(stage_map)
            plot_df.dropna(subset=['Growth_Stage_Desc'], inplace=True)

            if not plot_df.empty:
                
                grouped = plot_df.groupby('Growth_Stage_Desc')['Crop_Stress_Indicator']
                summary = grouped.agg(
                    median_stress='median',
                    q1=lambda s: s.quantile(0.25),
                    q3=lambda s: s.quantile(0.75),
                    count='count'
                ).reset_index()
                summary['iqr'] = summary['q3'] - summary['q1']
                
                high_stress_threshold = 70
                high_share = (
                    plot_df.assign(high=(plot_df['Crop_Stress_Indicator'] > high_stress_threshold))
                    .groupby('Growth_Stage_Desc')['high']
                    .mean()
                    .reindex(summary['Growth_Stage_Desc'])
                    .fillna(0)
                )
                summary['high_pct'] = (high_share.values * 100).round(1)
                
                summary['error_y'] = summary['q3'] - summary['median_stress']
                
                order = ['1. Germination', '2. Vegetative', '3. Flowering', '4. Grain Filling/Fruiting']
                summary['Growth_Stage_Desc'] = pd.Categorical(summary['Growth_Stage_Desc'], categories=order, ordered=True)
                summary = summary.sort_values('Growth_Stage_Desc')
                
                most_vulnerable_stage = summary.loc[summary['median_stress'].idxmax(), 'Growth_Stage_Desc']
                bar_colors = ['#A6CEE3' if stage != most_vulnerable_stage else '#E31A1C' for stage in summary['Growth_Stage_Desc']]
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=summary['Growth_Stage_Desc'],
                    y=summary['median_stress'],
                    error_y=dict(type='data', array=summary['error_y'], visible=True, thickness=1.2, color='rgba(0,0,0,0.5)'),
                    marker_color=bar_colors,
                    hovertemplate="<b>%{x}</b><br>Median Stress: %{y:.1f}<br>High Stress (>70): %{customdata:.1f}%<extra></extra>",
                    customdata=summary['high_pct']
                ))
                
                fig.update_traces(text=[f"{p:.1f}%" for p in summary['high_pct']], textposition='outside')
                fig.update_layout(
                    title=f'Stress by Growth Stage for {selected_crop_growth} (median ± IQR upper)',
                    xaxis_title='Crop Growth Stage',
                    yaxis_title='Crop Stress Indicator (0-100)',
                    uniformtext_minsize=10,
                    uniformtext_mode='hide',
                    showlegend=False
                )
                
                fig.add_annotation(
                    x=most_vulnerable_stage,
                    y=float(summary.loc[summary['Growth_Stage_Desc'] == most_vulnerable_stage, 'median_stress']),
                    text="Most Vulnerable",
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-40,
                    bgcolor="rgba(255,255,255,0.7)"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Dynamic crop-specific explanation
                mv_stage_str = str(most_vulnerable_stage)
                mv_median = float(summary.loc[summary['Growth_Stage_Desc'] == most_vulnerable_stage, 'median_stress'])
                mv_high_pct = float(summary.loc[summary['Growth_Stage_Desc'] == most_vulnerable_stage, 'high_pct'])
                mv_q3 = float(summary.loc[summary['Growth_Stage_Desc'] == most_vulnerable_stage, 'q3'])
                mv_q1 = float(summary.loc[summary['Growth_Stage_Desc'] == most_vulnerable_stage, 'q1'])
                mv_iqr = mv_q3 - mv_q1
                
                crop_reason = {
                    'Wheat': (
                        "The reproductive phase (especially Flowering–Grain Filling) in wheat is highly sensitive to heat "
                        "and water deficit. Stress at this stage disrupts pollination and grain filling."
                    ),
                    'Rice': (
                        "Rice becomes vulnerable from late vegetative into flowering due to its need for stable water levels "
                        "and high humidity; poor aeration or hot, dry winds elevate stress and disease pressure."
                    ),
                    'Maize': (
                        "Maize is most vulnerable at tasseling/silking (Flowering). Short water or heat stress during this window "
                        "reduces kernel set and spikes the stress indicator."
                    )
                }.get(selected_crop_growth, "Reproductive phases are generally the most sensitive due to flower and seed formation.")
                
                st.subheader("Why is this stage most vulnerable?")
                st.markdown(f"""
                - **Crop**: `{selected_crop_growth}`
                - **Most Vulnerable Stage**: `{mv_stage_str}`
                - **Median Stress**: `{mv_median:.1f}`
                - **IQR (Q1–Q3)**: `{mv_q1:.1f} – {mv_q3:.1f}` (width `{mv_iqr:.1f}`)
                - **High-Stress Share (>70)**: `{mv_high_pct:.1f}%`
                
                **Agronomic rationale:** {crop_reason}
                """)
            else:
                st.warning(f"No data available for the growth stages of {selected_crop_growth}.")
        else:
            st.warning(f"No data available for the selected crop: {selected_crop_growth}.")

    st.subheader("Professional Insights & Recommendations")
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **Key Insight (data-driven):**
        The highlighted red bar is the stage with the highest median stress. Use the labels above bars to read the share of >70 stress, and the upper IQR error bar to judge tail risk. 
        Together, these show which stage is most vulnerable for the currently selected crop and how frequently severe stress occurs at that stage.
        """)
        st.success("""
        **Potential Benefit:**
        Prioritize resources (e.g., supplemental irrigation, heat/wind protection, phase-timed nutrition) in the identified vulnerable window. This aligns interventions with risk peaks and improves the **resource-to-yield conversion ratio**.
        """)
    with col2:
        st.warning("""
        **AI & ML Potential:**
        A **dynamic Bayesian network** or a **sequential model (like LSTM)** could be used to model the transitions between growth stages and predict the probability of entering a high-stress state at the next stage, given the current environmental conditions.
        """)
        st.error("""
        **Potential Risk:**
        Stage labelling and timing can vary by field conditions, and some stages may have fewer samples. Interpret small-sample stages carefully and consider smoothing or pooling adjacent stages when needed.
        """)

# Case 9: Soil Health Management
elif page == "9. Soil Health Management":
    st.header("9. Soil Health Management")
    st.markdown("""
    **Problem:** Soil health is the foundation of agriculture, but its key components are often managed in isolation. How do pH and organic matter jointly impact overall crop health?
    """)

    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Filters")
        ph_bins = st.slider("Number of pH Bins", 5, 30, 15)
        om_bins = st.slider("Number of Organic Matter Bins", 5, 30, 15)
        healthy_threshold = st.slider("Highlight cells with Healthy rate ≥", 50, 95, 70, step=5, key="soil_health_rate_threshold")

        st.subheader("Analysis")
        st.markdown("""
        This heatmap shows the likelihood of being **Healthy** for each pH–OM cell (share of Healthy per cell).
        Cyan overlay highlights cells above the chosen healthy-rate threshold, and contour lines help reveal boundaries.
        """)

    with col2:
        map_df = df[['Soil_pH', 'Organic_Matter', 'Crop_Health_Label']].dropna().copy()
        if len(map_df) == 0:
            st.warning("No data available to compute heatmaps.")
        else:
            map_df['ph_bin'] = pd.cut(map_df['Soil_pH'], bins=ph_bins)
            map_df['om_bin'] = pd.cut(map_df['Organic_Matter'], bins=om_bins)
            agg = map_df.groupby(['ph_bin', 'om_bin'])['Crop_Health_Label'].mean().reset_index(name='healthy_rate')
            def bin_center(interval):
                try:
                    return (interval.left + interval.right) / 2
                except Exception:
                    return np.nan
            agg['ph_center'] = agg['ph_bin'].apply(bin_center)
            agg['om_center'] = agg['om_bin'].apply(bin_center)
            agg = agg.dropna(subset=['ph_center', 'om_center'])
            if len(agg) == 0:
                st.warning("Insufficient binned data to visualize.")
            else:
                hm = go.Figure()
                hm.add_trace(go.Heatmap(
                    x=agg['ph_center'],
                    y=agg['om_center'],
                    z=agg['healthy_rate'],
                    colorscale='RdYlGn',
                    zmin=0, zmax=1,
                    colorbar_title='Healthy rate'
                ))
                thr = healthy_threshold / 100.0
                mask = (agg['healthy_rate'] >= thr).astype(int)
                overlay_colorscale = [
                    [0.0, 'rgba(0,0,0,0)'],
                    [0.5, 'rgba(0,0,0,0)'],
                    [1.0, 'rgba(0,255,255,0.35)']
                ]
                hm.add_trace(go.Heatmap(
                    x=agg['ph_center'],
                    y=agg['om_center'],
                    z=mask,
                    colorscale=overlay_colorscale,
                    showscale=False
                ))
                hm.add_trace(go.Contour(
                    x=agg['ph_center'],
                    y=agg['om_center'],
                    z=agg['healthy_rate'],
                    contours=dict(start=0.4, end=0.9, size=0.1, coloring='none'),
                    line=dict(color='black', width=1),
                    showscale=False,
                    hoverinfo='skip'
                ))
                hm.update_layout(
                    title=f'Soil Health Likelihood Map (Healthy rate; cyan = ≥{healthy_threshold}%)',
                    xaxis_title='Soil pH (bin center)',
                    yaxis_title='Organic Matter (%) (bin center)'
                )
                st.plotly_chart(hm, use_container_width=True)

    st.subheader("Professional Insights & Recommendations")
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **Key Insight:**
        The healthy-rate heatmap shows a consistent pattern: **near‑neutral pH (≈6–7) combined with OM ≥ ~3%** most frequently falls in green areas and the cyan overlay (≥ threshold). 
        In contrast, **extreme pH** (too acidic or too alkaline) yields a lower healthy rate even when OM is high. The tight contour lines around pH ~5.5–6.0 and ~7.5–8.0 indicate **sharp transition boundaries** between soil profiles that are more likely to be healthy versus unhealthy.
        """)
        st.success("""
        **Potential Benefit:**
        Management can focus on **shifting field pH–OM combinations into the cyan zone**:
        1) perform **pH correction** (e.g., liming acidic soils) toward 6–7;
        2) gradually raise **OM to ≥3%** (compost/manure/cover crops).
        This strategy unlocks nutrient availability and increases the likelihood of fields sitting in the “high healthy‑rate” region.
        """)
    with col2:
        st.warning("""
        **AI & ML Potential:**
        Build a **classification/probabilistic model** (e.g., Gradient Boosting/Logistic) to predict healthy rate from pH and OM, then produce **action recommendations** (e.g., lime amount / OM addition) to move a pH–OM cell over a target threshold (e.g., 70%). 
        Use SHAP/partial dependence to quantify key contour boundaries and refresh them as new data arrives.
        """)
        st.error("""
        **Potential Risk:**
        The heatmap depends on **binning choices** and **data density per cell**; sparse cells can yield unstable estimates. 
        Mitigations: choose reasonable bin counts, consider a **minimum per‑cell count** for interpretation, and validate patterns with smoothing or complementary analyses.
        """)

# Case 10: Climate Impact on Crops
elif page == "10. Climate Impact on Crops":
    st.header("10. Climate Impact on Crops")
    st.markdown("""
    **Problem:** Different crops respond differently to climatic conditions. How does productivity vary for each crop type under different rainfall scenarios?
    """)

    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Filters")
        selected_crops_climate = st.multiselect("Select Crop Types", df['Crop_Type'].unique(), default=df['Crop_Type'].unique(), key="climate_crop")
        
        st.subheader("Analysis")
        st.markdown("""
        This **Box Plot** compares yield distribution by crop type across different rainfall levels. The progressive color scale indicates rainfall intensity. Hover over boxes for detailed stats (median, IQR, etc.).
        """)

    with col2:
        plot_df = df[df['Crop_Type'].isin(selected_crops_climate)]
        fig = px.box(
            plot_df,
            x='Crop_Type',
            y='Expected_Yield',
            color='Rainfall_Category',
            title='Yield Distribution by Crop Type and Rainfall Level',
            color_discrete_map={ 
                'Low': '#E45756', 
                'Medium': '#F28E2B', 
                'High': '#4C78A8' 
            },
            labels={
                "Crop_Type": "Crop Type",
                "Expected_Yield": "Expected Yield (kg/ha)",
                "Rainfall_Category": "Rainfall Level"
            }
        )
        fig.update_traces(quartilemethod="exclusive") 
        st.plotly_chart(fig, use_container_width=True)

        summary = (plot_df
                   .groupby(['Crop_Type', 'Rainfall_Category'])['Expected_Yield']
                   .median()
                   .unstack('Rainfall_Category'))
        if summary is not None and len(summary) > 0:
            st.markdown("Median Expected Yield (kg/ha) by Rainfall Level")
            st.dataframe(summary.fillna(0).round(0))
            auto_notes = []
            detailed_notes = []
            for crop, row in summary.iterrows():
                low = row.get('Low', np.nan)
                med = row.get('Medium', np.nan)
                high = row.get('High', np.nan)
                parts = []
                if not np.isnan(low) and not np.isnan(med):
                    parts.append("lower at Low than Medium" if low < med else "not lower at Low than Medium")
                if not np.isnan(low) and not np.isnan(high):
                    parts.append("lower at Low than High" if low < high else "not lower at Low than High")
                trend = ", ".join(parts) if parts else "insufficient data"
                auto_notes.append(f"- {crop}: {trend}.")
                vals = {k: v for k, v in [('Low', low), ('Medium', med), ('High', high)] if not np.isnan(v)}
                if vals:
                    best_cat = max(vals, key=vals.get)
                    diffs = []
                    if not np.isnan(low) and not np.isnan(med) and low != med:
                        diffs.append(f"Low vs Medium: ~{(med - low):.0f} kg/ha")
                    if not np.isnan(low) and not np.isnan(high) and low != high:
                        diffs.append(f"Low vs High: ~{(high - low):.0f} kg/ha")
                    detailed_notes.append(f"- {crop}: best at {best_cat}. " + ("; ".join(diffs) if diffs else ""))
            if auto_notes:
                st.markdown("Yield comparison (data-driven):")
                st.markdown("\n".join(auto_notes))
            climate_notes_md = "\n".join(detailed_notes) if detailed_notes else "Insufficient data for detailed differences."
        else:
            climate_notes_md = "Insufficient data to compute rainfall effects."

    st.subheader("Professional Insights & Recommendations")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"""
        **Key Insight (data-driven):**
        The per-crop median yields confirm the rainfall response patterns shown above.
        Use these notes to verify whether Low rainfall truly underperforms for each crop:
        {climate_notes_md}
        """)
        st.success("""
        **Potential Benefit:**
        Translate the findings into **rainfall-aware crop planning**. Where Low rainfall depresses yield for a crop, prioritize drought-tolerant varieties or schedule supplemental irrigation; where Medium/High is consistently best, align planting windows and water allocation to those regimes.
        """)
    with col2:
        st.warning("""
        **AI & ML Potential:**
        By combining this data with long-term climate model projections (e.g., from IPCC reports), a **Simulation Model** can be built to forecast the most profitable and viable crop mix for a specific region over the next 10-20 years, guiding strategic investment and land use policy.
        """)
        st.error("""
        **Potential Risk:**
        This analysis only considers total rainfall. The timing of the rainfall (e.g., during planting vs. harvesting) is just as important. A more granular, time-series analysis is needed. Furthermore, the impact of rainfall is mediated by irrigation, which is another layer of complexity.
        """)
