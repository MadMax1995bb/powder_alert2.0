# **PowderAlert 2.0: Live Powder Day Forecaster**

## **Project Overview**  
PowderAlert 2.0 is a **freeride planning tool** designed for ski and snowboard enthusiasts, predicting **snow depth**, **temperature**, and **wind speed** at the Austrian skiing resort **Hochfügen**. By combining historical weather data and advanced machine learning techniques, the app helps users identify optimal powder days to avoid disappointment on the slopes.

Predictions are made using **deep learning models** and integrated into a **Powder Day Indicator** that evaluates conditions for up to **48 hours in advance**.

---

## **Key Features**  
- **Forecast Models**:
  - **Snow Depth**: Predicted using a **Darts TransformerModel** with hyperparameter optimization.
  - **Temperature and Wind Speed**: Predicted via **LSTM deep learning models**.
- **Powder Day Indicator**: Combines predictions of snow depth, temperature, and wind speed into a user-friendly score for freeride conditions.
- **Interactive App**: Visualizes forecasts and allows users to plan their skiing or snowboarding trips.

---

## **Data Sources**  
The project leverages historical weather data from the **[Open-Meteo API](https://open-meteo.com/en/docs/historical-weather-api)**, which provided the most consistent and usable data for our models. 

We initially considered incorporating the **OpenWeatherMap API**, but its data format did not match the requirements for seamless integration, so it was excluded from the final implementation.

### **Features**  
The dataset includes 23 features, such as:  
- Temperature (min/max)  
- Snowfall  
- Wind speed  
- Weather conditions and descriptions  

Data preprocessing involved cleaning, removing outliers, and aggregating data into meaningful time intervals.

---

## **Modeling Approach**  
Our models capture **seasonal weather patterns** and deliver reliable forecasts, especially in stable conditions:  
1. **Darts TransformerModel**: Optimized with Optuna for snow depth prediction.  
   - Metrics: Mean Absolute Error (MAE), backtesting, and historical forecasting.  
2. **LSTM Deep Learning Models**: Predict temperature and wind speed with fine-tuned hyperparameters.  

---

## **The App**  
[Explore the PowderAlert 2.0 App](https://powderalert.streamlit.app/)  

The app offers:  
- **Real-Time Forecasts**: Access accurate predictions for snow depth, wind speed, and temperature.  
- **Powder Day Indicator**: Simplifies decision-making for freeride enthusiasts.  
- **Scalable Design**: Built for future enhancements, including additional locations and avalanche data integration.

---

## **Future Scalability**  
- Add more skiing locations and weather stations.  
- Incorporate avalanche bulletins and NLP-based analyses.  
- Regularly retrain models with updated weather data.  
- Enhance the user interface with interactive maps and advanced customization options.  
- Deploy a mobile-friendly version of the app.  

For detailed ideas, check out our [Trello Board](https://trello.com/b/ZE5LyJcF/powderalert20).

---

## **Acknowledgments**  
This project was developed as part of the **Le Wagon AI & Data Science Bootcamp (Batch #1871)** by:  
- Max Burger  
- Anita Geiss  
- Wisal Dhakouani  
- Torsten Wrigley  
- Luis Spee  

We thank the instructors and the Le Wagon community for their guidance and support.

---

## **Connect**  
For questions or further collaboration, connect with me on:  
- **LinkedIn**: [Maximilian Andreas Burger](https://www.linkedin.com/in/maximilian-andreas-burger/)  
- **Slack**: Max Burger (maxburger95@gmx.de)
