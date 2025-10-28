# 🌾 Precision Farming Analytics Dashboard

## Overview
This interactive dashboard, built with Streamlit, provides a comprehensive analysis of agricultural data to support precision farming decisions. It features 10 key analysis cases, transforming raw data into actionable insights for farmers, researchers, and agronomists.

1.  **Crop Health Diagnostics**: Distinguish healthy vs. unhealthy crops using remote sensing data (NDVI & Chlorophyll).
2.  **Pest Outbreak Management**: Quantify the economic impact of pest damage and identify vulnerable crops.
3.  **Yield Optimization Factors**: Identify environmental and soil factors with the strongest correlation to yield.
4.  **Precision Irrigation Strategy**: Determine the optimal soil moisture range for different crops.
5.  **Environmental Stress Analysis**: Analyze how interacting environmental factors contribute to crop stress.
6.  **Disease Risk Assessment**: Identify high-risk temperature and humidity "danger zones" for disease.
7.  **Resource Use Efficiency**: Visualize the combined impact of soil pH and organic matter on yield.
8.  **Growth Stage Vulnerability**: Pinpoint the most critical growth stages vulnerable to stress.
9.  **Soil Health Management**: Understand the joint impact of soil properties on overall crop health.
10. **Climate Impact on Crops**: Assess crop productivity variations under different rainfall scenarios.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+

### Setup Instructions

1.  **Clone the repository or download the source code.**

2.  **Navigate to the project directory:**
    ```bash
    cd /path/to/your/project
    ```

3.  **Install the required packages using `requirements.txt`:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run crop_health_dashboard.py
    ```
    The dashboard will open in your default web browser.

## 📊 Features

-   **Interactive Filtering**: Dynamically filter data by Crop Type, stress thresholds, and other parameters for customized analysis.
-   **Advanced Visualizations**: Utilizes Plotly for a rich variety of charts, including 3D scatter plots, density heatmaps, violin plots, and faceted views.
-   **Statistical Validation**: Integrates statistical tests (e.g., t-test, Mann-Whitney U) to validate visual findings.
-   **Professional Insights**: Each analysis case is accompanied by expert interpretations, potential benefits, AI/ML applications, and associated risks.
-   **Data Caching**: Employs Streamlit's caching for optimized performance and fast load times.

## 💾 Data Source
The dashboard currently loads the dataset directly from a public Google Drive URL. No local `.csv` file is needed. The expected dataset includes columns such as:
-   `Crop_Type`
-   `Crop_Health_Label`
-   `NDVI`
-   `Chlorophyll_Content`
-   `Pest_Damage`
-   `Expected_Yield`
-   `Soil_Moisture`
-   `Temperature`
-   `Humidity`
-   `Soil_pH`
-   `Organic_Matter`
-   and other environmental indicators.

## Contributing
Feel free to submit issues and enhancement requests. All contributions are welcome!
