# **PowderAlert 2.0: Snow Quality and Depth Prediction**

## **Project Overview**  
The goal of this project is to predict **snow quality** and **snow depth** at the Austrian skiing resort **Hochfügen** using historical weather data. Predictions are made via regression and classification tasks. These results are combined into a scoring model that evaluates how good a powder day might be for a selected timeframe (up to two days in the future).  

### **Scope**  
Currently, the project scope is limited due to a paywall restricting access to historical bulk weather data. However, predictions leverage:  
- **Regression Tasks**: Predicting snow depth, wind speed, and soil temperature.  
- **Classification Tasks** (optional): Predicting weather conditions (e.g., weather code).  
- **Scoring Model**: Evaluating the skiing conditions for powder days based on the predicted results.  

---

## **Datasets**  
We use two main datasets for this project, combining them to enrich the base dataset:  
1. **Base Dataset**: [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)  
2. **Additional Dataset**: [OpenWeatherMap Historical Bulk Data](https://openweathermap.org/history-bulk)  

### **Merged Parameters**  
The following parameters are integrated into the base dataset from OpenWeatherMap:  
- City Name  
- `main.temp_min`, `main.temp_max`  
- Rain (`rain_3h`)  
- Weather Code (`weather.id`)  
- Weather Description (`weather.main`, `weather.description`)  

---

## **Project Workflow**  
The overall project process is illustrated in the diagram below:  
![Project Overview](https://github.com/user-attachments/assets/145c1ebb-443d-43d6-87a1-f7be001f322b)  
*(This diagram might be updated as the project evolves.)*  

Key milestones during the first phase of the project are documented here:  
![Phase 1 Steps](https://github.com/user-attachments/assets/d4893370-aa62-4292-8f21-e8def598a2b2)  

For detailed progress tracking, check out our Trello board:  
👉 [Trello: PowderAlert 2.0](https://trello.com/b/ZE5LyJcF/powderalert20)  

---

## **Models**  
We focus on time-series forecasting using hourly data. Our primary model is based on the **[DARTS library](https://unit8co.github.io/darts/)**, which supports advanced time-series models.  

### **Planned Models**  
- **Regression**:  
  - Snow depth for time slots (up to 48 hours ahead).  
  - Wind speed.  
  - Soil temperature.  
- **Classification** (optional): Predicting weather conditions (e.g., weather code).  

### **Live Data**  
We integrate real-time data from the following APIs:  
- [OpenWeatherMap API](https://openweathermap.org/api)  
- [Open-Meteo API](https://open-meteo.com/en/docs)  

---

## **Final Interface Requirements**  
To deliver a user-friendly interface, we aim for:  
- A working API.  
- Dockerized deployment.  
- Cloud-run compatibility.  
- A web interface.  

### **Optional Features**  
- Interactive map to select locations or input latitude/longitude.  
- Sliders to set forecast times (e.g., **6 AM next day**, **12 PM next day**, etc.).  
- Additional user-friendly features.  

---

## **Future Scope**  
The project can be extended with additional data and functionality, such as:  
- Incorporating more datasets for broader coverage.  
- Adding **avalanche bulletin reports** via web scraping and **NLP models**.  
- Improving the web interface or deploying a polished app.  

---  
