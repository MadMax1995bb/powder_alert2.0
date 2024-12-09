Scope of the project:
The goal of this project is to predict snow quality and snow height in the Austrian skiing resort Hochfügen based on historical weather data.
The scope unfortunately is limited due to a paywall for historical bulk data.
Modeling is done in both regression tasks and one classification task.
The combination of the prediction results lead into a scoring model for how good the powder day might be.

Datasets:
https://open-meteo.com/en/docs/historical-weather-api (Base dataset)
Data: https://openweathermap.org/history-bulk (Additional dataset merged on the other one)

The following parameters where merged from the additional dataset into the base dataset:
- city name/ main.temp_min/ main.temp_max/ rain_3h/ weatehr.id/ weather.main/ weather.description

Live data for the predictions is available here:
https://openweathermap.org/api
https://open-meteo.com/en/docs

Here you can find a project overview:
![grafik](https://github.com/user-attachments/assets/145c1ebb-443d-43d6-87a1-f7be001f322b)

Steps in our Data structuring and cleaning are:
- Data sourcing
- Data merging
- Data cleaning
- Exploratory Data Analysis

Steps in our Feature engineering


Models:
We are modeling mainly two regression task and one classification task.
DART
The data for prediction comes form the two API´s and needs to go throught the preprossesing pipeline ase well after it was merged.


Requirements for the final interface are:
- functional API
- Running via Doker
- Ability to cloud run
- Web interface

- Including a mape where we can click on the location/ input lat/lon (optional)
- Include some fancy emojis
- add sliders to change the time, we want to forecasts for (e.g. 6am next day, 12am next day, 6am the day after tomorrow)
- etc.
