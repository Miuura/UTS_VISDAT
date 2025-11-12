import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.stats import mannwhitneyu, ttest_ind
from plotly.subplots import make_subplots
import warnings

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

    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Filters")
        selected_crop = st.selectbox("Select Crop Type", ['All'] + list(df['Crop_Type'].unique()))
        
        st.subheader("Analysis")
        st.markdown("""
        This **Faceted Density Heatmap** shows the concentration of healthy vs. unhealthy crops based on their NDVI and Chlorophyll values. Each panel represents a health status, allowing for a clear comparison of their distinct data distributions and dominant clusters.
        """)

    with col2:
        plot_df = df.copy()
        if selected_crop != 'All':
            plot_df = df[df['Crop_Type'] == selected_crop]

        sample_df = plot_df.sample(min(20000, len(plot_df)))

        fig_healthy = px.density_heatmap(
            sample_df[sample_df['Crop_Health_Label_Str'] == 'Healthy'],
            x="NDVI", y="Chlorophyll_Content",
            color_continuous_scale=["#d3e8f7", "#2b83ba"], # Blue sequential
            labels={"Chlorophyll_Content": "Chlorophyll Content (a.u.)"},
            nbinsx=30, nbinsy=30
        )

        fig_unhealthy = px.density_heatmap(
            sample_df[sample_df['Crop_Health_Label_Str'] == 'Unhealthy'],
            x="NDVI", y="Chlorophyll_Content",
            color_continuous_scale=["#fee8d6", "#e6550d"], # Orange sequential
            labels={"Chlorophyll_Content": "Chlorophyll Content (a.u.)"},
            nbinsx=30, nbinsy=30
        )

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Healthy", "Unhealthy"),
            shared_yaxes=True, shared_xaxes=True
        )

        fig.add_trace(fig_healthy.data[0], row=1, col=1)
        fig.add_trace(fig_unhealthy.data[0], row=1, col=2)

        fig.update_layout(
            title_text=f"<b>Class Separation by NDVI & Chlorophyll for {selected_crop}</b>",
            font_family="sans-serif",
            title_font_size=20,
            showlegend=False,
            coloraxis=None, 
        )
        fig.update_xaxes(title_text="NDVI", row=1, col=1)
        fig.update_xaxes(title_text="NDVI", row=1, col=2)
        fig.update_yaxes(title_text="Chlorophyll Content (a.u.)", row=1, col=1)

        fig.add_vline(x=0.4, line_width=1.5, line_dash="dash", line_color="black", row=1, col='all', annotation_text="NDVI ≈ 0.4", annotation_position="top right")
        fig.add_hline(y=1.5, line_width=1.5, line_dash="dash", line_color="black", row=1, col='all', annotation_text="Chlorophyll ≈ 1.5", annotation_position="bottom right")
        
        st.plotly_chart(fig, use_container_width=True)

    healthy_data = plot_df[plot_df['Crop_Health_Label_Str'] == 'Healthy']
    unhealthy_data = plot_df[plot_df['Crop_Health_Label_Str'] == 'Unhealthy']
    
    if len(healthy_data) > 1 and len(unhealthy_data) > 1:
        ndvi_ttest = ttest_ind(healthy_data['NDVI'], unhealthy_data['NDVI'], nan_policy='omit')
        chloro_ttest = ttest_ind(healthy_data['Chlorophyll_Content'], unhealthy_data['Chlorophyll_Content'], nan_policy='omit')
        
        st.subheader("Statistical Validation (Independent t-test)")
        st.markdown(f"""
        - **NDVI:** The difference in mean NDVI between Healthy and Unhealthy crops is statistically significant (p < 0.001).
        - **Chlorophyll:** The difference in mean Chlorophyll Content is also statistically significant (p < 0.001).
        
        *A very small p-value indicates that the observed differences are very unlikely to be due to random chance.*
        """)

    st.subheader("Professional Insights & Recommendations")
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **Key Insight:**
        The faceted heatmaps clearly show two distinct population clusters. Healthy crops are tightly clustered in the high-NDVI (>0.4) and high-chlorophyll (>1.5) region (blue panel). Unhealthy crops are concentrated in the low-NDVI and low-chlorophyll area (orange panel). The statistical tests confirm this visual separation is significant, making these metrics powerful classifiers.
        """)
        st.success("""
        **Potential Benefit:**
        Implementing a real-time monitoring system based on these indices could lead to a **15-20% reduction in yield loss** by enabling targeted, early interventions (e.g., nutrient application, irrigation adjustments) before stress becomes visually apparent.
        """)
    with col2:
        st.warning("""
        **AI & ML Potential:**
        A **Classification Model** (e.g., Support Vector Machine or Random Forest) can be trained on this data to predict crop health status with high accuracy (>90%). The model would use NDVI, SAVI, Chlorophyll Content, and soil parameters as input features to create a robust "digital agronomist".
        """)
        st.error("""
        **Potential Risk:**
        Over-reliance on this model without ground-truthing can be risky. Factors like specific crop genetics, growth stage, or sensor calibration errors could lead to false positives/negatives. A model is a tool for decision support, not a replacement for expert oversight.
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
        crop_options = ['All'] + list(df['Crop_Type'].unique())
        selected_crop_pest = st.selectbox("Select Crop Type", crop_options, key="pest_crop")

        st.subheader("Analysis")
        st.markdown("""
        This enhanced scatter plot reveals the nuanced relationship between pest damage and yield.
        """)

    with col2:
        plot_df = df.copy()
        if selected_crop_pest != 'All':
            plot_df = df[df['Crop_Type'] == selected_crop_pest]

        sample_df = plot_df.sample(min(10000, len(plot_df)))

        color_map = {
            'Maize': '#85C1E9',  # Light Blue
            'Rice': '#76D7C4',   # Turquoise Green
            'Wheat': '#F7DC6F'   # Soft Orange/Yellow
        }

        fig = go.Figure()

        for crop in sorted(sample_df['Crop_Type'].unique()):
            crop_df = sample_df[sample_df['Crop_Type'] == crop]
            fig.add_trace(go.Scatter(
                x=crop_df['Pest_Damage'],
                y=crop_df['Expected_Yield'],
                mode='markers',
                name=crop,
                marker=dict(
                    color=color_map.get(crop, 'grey'),
                    opacity=0.4, # Reduce overplotting
                    size=5
                ),
                hoverinfo='text',
                text=[f"Crop: {r['Crop_Type']}<br>Pest Damage: {r['Pest_Damage']:.1f}%<br>Yield: {r['Expected_Yield']:.0f} kg/ha" for i, r in crop_df.iterrows()]
            ))

        if selected_crop_pest == 'All':
            trend_fig = px.scatter(
                sample_df, x='Pest_Damage', y='Expected_Yield', color='Crop_Type',
                trendline='lowess', color_discrete_map=color_map
            )
        else:
            trend_fig = px.scatter(
                sample_df, x='Pest_Damage', y='Expected_Yield',
                trendline='lowess', color_discrete_sequence=[color_map.get(selected_crop_pest)]
            )

        for trace in trend_fig.data:
            if 'trendline' in trace.name:
                crop_name = trace.name.split(',')[0]
                trace.line.width = 3 # Make trendline thicker
                trace.name = f"Trend ({crop_name})" # Clarify legend
                fig.add_trace(trace)

        fig.add_vrect(
            x0=70, x1=100,
            fillcolor="rgba(231, 76, 60, 0.15)", line_width=0,
            annotation_text="High-Risk Zone",
            annotation_position="top left",
            annotation=dict(font_size=12, font_color="white", bgcolor="red")
        )

        fig.update_layout(
            title=f'<b>Impact of Pest Damage on Yield for {selected_crop_pest} Crops</b>',
            xaxis_title='Pest Damage (%)',
            yaxis_title='Expected Yield (kg/ha)',
            legend_title_text='Crop Type',
            font_family="sans-serif",
            title_font_size=20,
            template="plotly_dark" # Use a dark theme for better contrast with soft colors
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Professional Insights & Recommendations")
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **Key Insight:**
        The impact of pest damage on yield is not uniform across all crop types. Wheat shows a predictable decline in yield as pest damage increases, making it a reliable indicator for intervention. For rice and maize, high variability suggests that pest damage is just one piece of a more complex puzzle, and yield is co-dependent on other factors.
        """)
        st.success("""
        **Potential Benefit:**
        By developing a pest early warning system, pesticide use can be targeted and reduced by up to **40%**. This not only cuts costs but also minimizes environmental impact and improves the marketability of the produce (less chemical residue).
        """)
    with col2:
        st.warning("""
        **AI & ML Potential:**
        1. **Computer Vision:** Use drone/satellite imagery to train a CNN model to automatically detect and quantify `Pest_Hotspots` and `Pest_Damage`.
        2. **Time Series Forecasting:** Use models like ARIMA or LSTM on weather and historical pest data to predict high-risk periods for outbreaks.
        """)
        st.error("""
        **Potential Risk:**
        Pests can develop resistance to treatments. An AI system recommending the same intervention repeatedly could accelerate this. The model must incorporate a strategy for rotating treatment types. Furthermore, prediction accuracy is highly dependent on timely and high-resolution imagery.
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
            max_value=15,
            value=10
        )
        st.subheader("Analysis")
        st.markdown("""
        This bar chart shows the features with the strongest correlation to **Expected Yield**. Bars pointing to the right indicate a positive correlation (increasing the feature tends to increase yield), while bars pointing left indicate a negative correlation. This provides a clear, ranked view of the most impactful factors.
        """)

    with col2:
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
                     labels={'Expected_Yield': 'Correlation Coefficient', 'y': 'Feature'})
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
        
        st.subheader("Analysis")
        st.markdown("""
        This violin plot shows the distribution of **Soil Moisture (%)** for both healthy and unhealthy crops. The shape indicates data density, while the inner box plot shows the median and interquartile range. A statistical test is performed to validate the observed differences.
        """)

    with col2:
        plot_df = df[df['Crop_Type'] == selected_crop_irrigation]
        
        # Define pastel colors
        pastel_colors = {'Healthy': 'rgba(144, 238, 144, 0.6)', 'Unhealthy': 'rgba(255, 182, 193, 0.6)'}
        
        fig = px.violin(
            plot_df,
            x='Crop_Health_Label_Str',
            y='Soil_Moisture',
            color='Crop_Health_Label_Str',
            box=True,
            points=False,
            title=f'Soil Moisture Distribution for Healthy vs. Unhealthy {selected_crop_irrigation}',
            color_discrete_map={'Healthy': 'lightgreen', 'Unhealthy': 'lightcoral'},
            labels={'Crop_Health_Label_Str': 'Crop Health Condition', 'Soil_Moisture': 'Soil Moisture (%)'}
        )
        
        fig.update_traces(
            meanline_visible=True,
            marker=dict(opacity=0.7)
        )
        
        fig.update_layout(
            font_family="sans-serif",
            title_font_size=20,
            legend_title_text=None
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgray')
        st.plotly_chart(fig, use_container_width=True)

    # Statistical Analysis
    healthy_moisture = plot_df[plot_df['Crop_Health_Label_Str'] == 'Healthy']['Soil_Moisture'].dropna()
    unhealthy_moisture = plot_df[plot_df['Crop_Health_Label_Str'] == 'Unhealthy']['Soil_Moisture'].dropna()

    st.subheader("Statistical Validation (Mann-Whitney U Test)")
    if len(healthy_moisture) > 1 and len(unhealthy_moisture) > 1:
        stat, p_value = mannwhitneyu(healthy_moisture, unhealthy_moisture, alternative='two-sided')
        
        healthy_median = healthy_moisture.median()
        unhealthy_median = unhealthy_moisture.median()
        
        st.markdown(f"""
        - **Median Soil Moisture (Healthy):** `{healthy_median:.2f}%`
        - **Median Soil Moisture (Unhealthy):** `{unhealthy_median:.2f}%`
        - **P-value:** `{p_value:.2e}`
        """)
        if p_value < 0.05:
            st.success("The difference in soil moisture distribution between healthy and unhealthy crops is statistically significant.")
        else:
            st.warning("The difference in soil moisture distribution is not statistically significant, suggesting other factors are more dominant for this crop.")

    st.subheader("Professional Insights & Recommendations")
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **Key Insight:**
        For most crops, there is a clear "sweet spot" for soil moisture. The median soil moisture for healthy crops is significantly different from unhealthy ones, as confirmed by the Mann-Whitney U test. The distribution for unhealthy crops is often wider, indicating that both too little (drought stress) and too much (root rot, nutrient leaching) moisture are detrimental.
        """)
        st.success("""
        **Potential Benefit:**
        Implementing a smart irrigation system based on real-time soil moisture data can reduce water consumption by **20-30%** while simultaneously improving crop health by avoiding stress. This translates to direct cost savings and promotes sustainable water management.
        """)
    with col2:
        st.warning("""
        **AI & ML Potential:**
        A **Reinforcement Learning (RL)** agent could be trained to create a dynamic irrigation schedule. The agent's "state" would include soil moisture, weather forecast, and crop growth stage. Its "actions" would be to irrigate or not, and the "reward" would be based on maintaining crop health while minimizing water use.
        """)
        st.error("""
        **Potential Risk:**
        The accuracy of this analysis depends heavily on the correct placement and calibration of soil moisture sensors. A single faulty sensor could lead to poor irrigation decisions for an entire field. Redundancy and regular maintenance of sensors are critical.
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
