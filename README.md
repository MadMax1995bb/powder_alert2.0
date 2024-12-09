Scope of the project:
The goal of this project is to predict snow quality and snow height in the Austrian skiing resort Hochfügen based on historical weather data. The scope currently is limited due to a paywall for historical bulk data. The predictions are done via both regression and classification tasks. The combination of the prediction results lead into a final scoring model which represents how good the powder day might be for a selected timeframe (up to two days in the future).

Datasets for the project:
https://open-meteo.com/en/docs/historical-weather-api (Base dataset)
https://openweathermap.org/history-bulk (Additional dataset merged on the other base dataset)
The following parameters where merged into the base dataset:
city name/ main.temp_min/ main.temp_max/ rain_3h/ weatehr.id/ weather.main/ weather.description

Here you can find a project overview:
![grafik](https://github.com/user-attachments/assets/145c1ebb-443d-43d6-87a1-f7be001f322b)
(Might be updated and adapted to our project construction)
Some important steps during the first project phase:
![grafik](https://github.com/user-attachments/assets/d4893370-aa62-4292-8f21-e8def598a2b2)

Important steps are documented on our Trello board here:
https://trello.com/b/ZE5LyJcF/powderalert20

Models:
For our modeling we are focussing mainly on a regression task which forecastes the snow depth for a given time slot the next couple of hours (up to two days). Furthermore (optional) we include a classification task (predicting a weather_code) and two further regression tasks (wind_speed and soil_temperature). We probably will use the DARTS model (https://unit8co.github.io/darts/) because we rely on hourly data/ Timeseries. The live data for our prediction comes form the following two API´s:
https://openweathermap.org/api
https://open-meteo.com/en/docs

Basic requirements for our final interface are:
- Working API
- Running via Doker
- Ability to cloud run
- Web interface
Further features we want to implement (optional): 
- Including a map where we can click on the location/ input lat/lon
- Add sliders to change the time, we want to forecasts for (e.g. 6am next day, 12am next day, 6am the day after tomorrow)
- Some other nice shit :)

On top of the final project outcome, the project scope could be further extended by e.g.
a) adding more data and thus extending the availability,
b) adding relevant Avalanche bulletin reports (via web scraping and using NLP-model) and finally
c) improving the webinterface/ deploying a even nicer App
