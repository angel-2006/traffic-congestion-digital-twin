# Traffic Congestion Prediction using Digital Twin

## Project Overview
This project aims to build a digital twin for traffic congestion prediction using:
- TomTom API for real-world traffic data
- SUMO for traffic simulation
- TraCI for simulation control
- Machine Learning for congestion prediction

## Project Goals
- Collect traffic flow data
- Predict congestion levels
- Simulate traffic conditions
- Compare real vs simulated traffic

## Folder Structure
- `data/` → raw and processed datasets
- `src/` → source code
- `sumo/` → SUMO simulation files
- `models/` → trained ML models
- `reports/` → outputs, plots, results

## Status
Project setup in progress.

## Environment Setup
A Python virtual environment is used for dependency management.

## Installed Libraries
- Data handling: pandas, numpy
- API access: requests, python-dotenv
- ML: scikit-learn, xgboost, lightgbm
- Visualization: matplotlib, seaborn, plotly, folium
- Geospatial: geopandas, shapely, osmnx
- Scheduling: schedule

## Step Completed
Implemented multi-location traffic data collection using TomTom Flow Segment Data API.

## Current Data Pipeline
- Reads predefined traffic observation points
- Fetches live traffic flow data for each point
- Extracts useful congestion-related features
- Saves output into CSV format

## Automated Data Collection
The traffic data pipeline now supports:
- repeated live traffic collection
- appending new records into a growing CSV dataset
- scheduled collection every 10 minutes