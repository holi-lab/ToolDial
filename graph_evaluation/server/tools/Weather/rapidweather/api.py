import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def call_5_day_3_hour_forecast_data_by_city_name(q: str, units: str=None, cnt: str=None, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can search weather forecast for 5 days with data every 3 hours by city name."
    q: City name, state code and country code divided by comma, use ISO 3166 country codes.
You can specify the parameter not only in English. In this case, the API response should be returned in the same language as the language of requested location name if the location is in our predefined list of more than 200,000 locations.
        units: Units of measurement. **standard**, **metric **and **imperial **units are available. If you do not use the **units **parameter, **standard **units will be applied by default.
        cnt: A number of timestamps, which will be returned in the API response.
        lang: You can use the **lang **parameter to get the output in your language
        
    """
    url = f"https://rapidweather.p.rapidapi.com/data/2.5/forecast"
    querystring = {'q': q, }
    if units:
        querystring['units'] = units
    if cnt:
        querystring['cnt'] = cnt
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidweather.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def call_5_day_3_hour_forecast_data_by_zip_code(zip: int, cnt: str=None, units: str=None, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Please note if country is not specified then the search works for USA as a default."
    zip: Zip code
        cnt: A number of timestamps, which will be returned in the API response.
        units: Units of measurement. **standard**, **metric **and **imperial **units are available. If you do not use the **units **parameter, **standard **units will be applied by default.
        lang: You can use the **lang **parameter to get the output in your language.
        
    """
    url = f"https://rapidweather.p.rapidapi.com/data/2.5/forecast"
    querystring = {'zip': zip, }
    if cnt:
        querystring['cnt'] = cnt
    if units:
        querystring['units'] = units
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidweather.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def current_air_pollution_data(lat: int, lon: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Current air pollution data.
		Besides basic Air Quality Index, the API returns data about polluting gases, such as Carbon monoxide (CO), Nitrogen monoxide (NO), Nitrogen dioxide (NO2), Ozone (O3), Sulphur dioxide (SO2), Ammonia (NH3), and particulates (PM2.5 and PM10).
		
		Air pollution forecast is available for 5 days with hourly granularity. Historical data is accessible from 27th November 2020."
    lat: Geographical coordinates (latitude, longitude)
        lon: Geographical coordinates (latitude, longitude)
        
    """
    url = f"https://rapidweather.p.rapidapi.com/data/2.5/air_pollution"
    querystring = {'lat': lat, 'lon': lon, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidweather.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def forecast_air_pollution_data(lat: int, lon: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Forecast air pollution data.
		Besides basic Air Quality Index, the API returns data about polluting gases, such as Carbon monoxide (CO), Nitrogen monoxide (NO), Nitrogen dioxide (NO2), Ozone (O3), Sulphur dioxide (SO2), Ammonia (NH3), and particulates (PM2.5 and PM10).
		
		Air pollution forecast is available for 5 days with hourly granularity. Historical data is accessible from 27th November 2020."
    
    """
    url = f"https://rapidweather.p.rapidapi.com/data/2.5/air_pollution/forecast"
    querystring = {'lat': lat, 'lon': lon, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidweather.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historical_air_pollution_data(start: int, lat: int, end: int, lon: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Historical air pollution data"
    start: Start date (unix time, UTC time zone), e.g. start=1606488670
        lat: Geographical coordinates (latitude, longitude)
        end: End date (unix time, UTC time zone), e.g. end=1606747870
        lon: Geographical coordinates (latitude, longitude)
        
    """
    url = f"https://rapidweather.p.rapidapi.com/data/2.5/air_pollution/history"
    querystring = {'start': start, 'lat': lat, 'end': end, 'lon': lon, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidweather.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def coordinates_by_location_name(q: str, limit: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Coordinates by location name"
    q: City name, state code (only for the US) and country code divided by comma. Please use ISO 3166 country codes.
        limit: Number of the locations in the API response (up to 5 results can be returned in the API response)
        
    """
    url = f"https://rapidweather.p.rapidapi.com/geo/1.0/direct"
    querystring = {'q': q, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidweather.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reverse_geocoding(lat: int, lon: int, limit: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Reverse geocoding allows to get name of the location (city name or area name) by using geografical coordinates (lat, lon). The **limit **parameter in the API call allows you to cap how many location names you will see in the API response."
    lat: Geographical coordinates (latitude, longitude)
        lon: Geographical coordinates (latitude, longitude)
        limit: Number of the location names in the API response (several results can be returned in the API response)
        
    """
    url = f"https://rapidweather.p.rapidapi.com/geo/1.0/reverse"
    querystring = {'lat': lat, 'lon': lon, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidweather.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def coordinates_by_zip_post_code(zip: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Coordinates by zip/post code"
    zip: Zip/post code and country code divided by comma. Please use ISO 3166 country codes
        
    """
    url = f"https://rapidweather.p.rapidapi.com/geo/1.0/zip"
    querystring = {'zip': zip, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidweather.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def one_call_api(lon: int, lat: int, lang: str=None, units: str=None, exclude: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The One Call API provides the following weather data for any geographical coordinates:
		
		- Current weather
		- Minute forecast for 1 hour
		- Hourly forecast for 48 hours
		- Daily forecast for 7 days
		- National weather alerts
		- Historical weather data for the previous 5 days"
    lon: Geographical coordinates (latitude, longitude)
        lat: Geographical coordinates (latitude, longitude)
        units: Units of measurement. **standard**, **metric **and **imperial **units are available. If you do not use the **units **parameter, **standard **units will be applied by default
        exclude: By using this parameter you can exclude some parts of the weather data from the API response. It should be a comma-delimited list (without spaces).
Available values:

- current
- minutely
- hourly
- daily
- alerts
        
    """
    url = f"https://rapidweather.p.rapidapi.com/data/2.5/onecall"
    querystring = {'lon': lon, 'lat': lat, }
    if lang:
        querystring['lang'] = lang
    if units:
        querystring['units'] = units
    if exclude:
        querystring['exclude'] = exclude
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidweather.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def call_5_day_3_hour_forecast_data_by_geographic_coordinates(lon: int, lat: int, units: str=None, lang: str=None, cnt: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can search weather forecast for 5 days with data every 3 hours by geographic coordinates."
    lon: Geographical coordinates (latitude, longitude)
        lat: Geographical coordinates (latitude, longitude)
        units: Units of measurement. **standard**, **metric **and **imperial **units are available. If you do not use the **units **parameter, **standard **units will be applied by default.
        lang: You can use the **lang **parameter to get the output in your language.
        cnt: A number of timestamps, which will be returned in the API response.
        
    """
    url = f"https://rapidweather.p.rapidapi.com/data/2.5/forecast"
    querystring = {'lon': lon, 'lat': lat, }
    if units:
        querystring['units'] = units
    if lang:
        querystring['lang'] = lang
    if cnt:
        querystring['cnt'] = cnt
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidweather.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def call_5_day_3_hour_forecast_data_by_city_id(is_id: int, lang: str=None, cnt: str=None, units: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can search weather forecast for 5 days with data every 3 hours by city ID. We recommend to call API by city ID to get unambiguous result for your city."
    id: City ID.
        lang: You can use the **lang **parameter to get the output in your language
        cnt: A number of timestamps, which will be returned in the API response.
        units: Units of measurement. **standard**, **metric **and **imperial **units are available. If you do not use the **units **parameter, **standard **units will be applied by default.
        
    """
    url = f"https://rapidweather.p.rapidapi.com/data/2.5/forecast"
    querystring = {'id': is_id, }
    if lang:
        querystring['lang'] = lang
    if cnt:
        querystring['cnt'] = cnt
    if units:
        querystring['units'] = units
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidweather.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cities_in_circle(lon: int, lat: int, cnt: int=None, lang: str=None, units: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API returns data from cities laid within definite circle that is specified by center point"
    lon: Geographical coordinates (latitude, longitude)
        lat: Geographical coordinates (latitude, longitude)
        cnt: Number of cities around the point that should be returned. The default number of cities is 5, the maximum is 50.
        lang: You can use the **lang **parameter to get the output in your language
        units: Units of measurement. **standard**, **metric **and **imperial **units are available. If you do not use the **units **parameter, **standard **units will be applied by default.
        
    """
    url = f"https://rapidweather.p.rapidapi.com/data/2.5/find"
    querystring = {'lon': lon, 'lat': lat, }
    if cnt:
        querystring['cnt'] = cnt
    if lang:
        querystring['lang'] = lang
    if units:
        querystring['units'] = units
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidweather.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def by_zip_code(zip: str, lang: str=None, units: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Please note if country is not specified then the search works for USA as a default."
    zip: Zip code
        lang: You can use the **lang **parameter to get the output in your language
        units: Units of measurement. **standard**, **metric **and **imperial **units are available. If you do not use the **units **parameter, **standard **units will be applied by default.
        
    """
    url = f"https://rapidweather.p.rapidapi.com/data/2.5/weather"
    querystring = {'zip': zip, }
    if lang:
        querystring['lang'] = lang
    if units:
        querystring['units'] = units
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidweather.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def by_geographic_coordinates(lon: int, lat: int, units: str=None, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "By geographic coordinates"
    lon: Geographical coordinates (latitude, longitude)
        lat: Geographical coordinates (latitude, longitude)
        units: Units of measurement. **standard**, **metric **and **imperial **units are available. If you do not use the **units **parameter, **standard **units will be applied by default.
        lang: You can use the **lang **parameter to get the output in your language
        
    """
    url = f"https://rapidweather.p.rapidapi.com/data/2.5/weather"
    querystring = {'lon': lon, 'lat': lat, }
    if units:
        querystring['units'] = units
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidweather.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def by_city_id(is_id: int, units: str=None, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can make an API call by city ID.
		We recommend to call API by city ID to get unambiguous result for your city."
    id: City ID. 
        units: Units of measurement. **standard**, **metric **and **imperial **units are available. If you do not use the **units **parameter, **standard **units will be applied by default.
        lang: You can use the **lang **parameter to get the output in your language
        
    """
    url = f"https://rapidweather.p.rapidapi.com/data/2.5/weather"
    querystring = {'id': is_id, }
    if units:
        querystring['units'] = units
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidweather.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def by_city_name(q: str, units: str=None, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can call by city name or city name, state code and country code. Please note that searching by states available only for the USA locations."
    q: City name, state code and country code divided by comma, Please, refer to [ISO 3166](https://www.iso.org/obp/ui/#search) for the state codes or country codes.
You can specify the parameter not only in English. In this case, the API response should be returned in the same language as the language of requested location name if the location is in our predefined list of more than 200,000 locations.
        units: Units of measurement. **standard**, **metric **and **imperial **units are available. If you do not use the **units **parameter, **standard **units will be applied by default.
        lang: You can use the **lang **parameter to get the output in your language
        
    """
    url = f"https://rapidweather.p.rapidapi.com/data/2.5/weather"
    querystring = {'q': q, }
    if units:
        querystring['units'] = units
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidweather.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def previous_5_days_historical_weather_data(dt: int, lon: int, lat: int, units: str=None, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Historical weather data for the previous 5 days"
    dt: Date from the previous five days (Unix time, UTC time zone), e.g. dt=1586468027
        lon: Geographical coordinates (latitude, longitude)
        lat: Geographical coordinates (latitude, longitude)
        units: Units of measurement. **standard**, **metric **and **imperial **units are available. If you do not use the **units **parameter, **standard **units will be applied by default.
        lang: You can use the **lang **parameter to get the output in your language
        
    """
    url = f"https://rapidweather.p.rapidapi.com/data/2.5/onecall/timemachine"
    querystring = {'dt': dt, 'lon': lon, 'lat': lat, }
    if units:
        querystring['units'] = units
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rapidweather.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

