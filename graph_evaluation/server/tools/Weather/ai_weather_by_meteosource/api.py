import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def historical_weather(date: str, lat: str='37.81021', place_id: str=None, units: str='auto', lon: str='-122.42282', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Receive **historical weather** data for a **given day** in the past **8 years**. Define your location using GPS coordinates or `place_id` from `Location endpoints`."
    date: The UTC day of the data in the past in `YYYY-MM-DD` format.
        lat: Latitude in format 12N, 12.3N, 12.3, or 13S, 13.2S, -13.4. **Alternatively, you can specify the location by parameter `place_id`.**
        place_id: Identifier of a place. To obtain the `place_id` for the location you want, please use `Location endpoints`. **Alternatively, you can specify the location by parameters `lat` and `lon`.**
        units: Unit system to be used. The available values are:

- `auto`: Select the system automatically, based on the forecast location.
- `metric`: Metric (SI) units (`°C`, `mm/h`, `m/s`, `cm`, `km`, `hPa`).
- `us`: Imperial units (`°F`, `in/h`, `mph`, `in`, `mi`, `Hg`).
- `uk`: Same as `metric`, except that visibility is in `miles` and wind speeds are in `mph`.
- `ca`: Same as `metric`, except that wind speeds are in `km/h` and pressure is in `kPa`.

        lon: Longitude in format 12E, 12.3E, 12.3, or 13W, 13.2W, -13.4. **Alternatively, you can specify the location by parameter `place_id`.**
        
    """
    url = f"https://ai-weather-by-meteosource.p.rapidapi.com/time_machine"
    querystring = {'date': date, }
    if lat:
        querystring['lat'] = lat
    if place_id:
        querystring['place_id'] = place_id
    if units:
        querystring['units'] = units
    if lon:
        querystring['lon'] = lon
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ai-weather-by-meteosource.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def weather_statistics(units: str='auto', place_id: str=None, lon: str='-122.42282', lat: str='37.81021', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get average weather: **long-term normals** for a given place for the next 30 days. Define your location using GPS coordinates or `place_id` from `Location endpoints`."
    units: Unit system to be used. The available values are:

- `auto`: Select the system automatically, based on the forecast location.
- `metric`: Metric (SI) units (`°C`, `mm/h`, `m/s`, `cm`, `km`, `hPa`).
- `us`: Imperial units (`°F`, `in/h`, `mph`, `in`, `mi`, `Hg`).
- `uk`: Same as `metric`, except that visibility is in `miles` and wind speeds are in `mph`.
- `ca`: Same as `metric`, except that wind speeds are in `km/h` and pressure is in `kPa`.

        place_id: Identifier of a place. To obtain the `place_id` for the location you want, please use `Location endpoints`. **Alternatively, you can specify the location by parameters `lat` and `lon`.**
        lon: Longitude in format 12E, 12.3E, 12.3, or 13W, 13.2W, -13.4. **Alternatively, you can specify the location by parameter `place_id`.**
        lat: Latitude in format 12N, 12.3N, 12.3, or 13S, 13.2S, -13.4. **Alternatively, you can specify the location by parameter `place_id`.**
        
    """
    url = f"https://ai-weather-by-meteosource.p.rapidapi.com/weather_statistics"
    querystring = {}
    if units:
        querystring['units'] = units
    if place_id:
        querystring['place_id'] = place_id
    if lon:
        querystring['lon'] = lon
    if lat:
        querystring['lat'] = lat
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ai-weather-by-meteosource.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hourly(lon: str='-122.42282', lat: str='37.81021', language: str='en', place_id: str=None, units: str='auto', timezone: str='auto', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Hourly weather** forecast for the next 5 days. **Global** data are based on our **AI technology**, which uses many different models. **Define your location** using GPS coordinates or `place_id` from `Location endpoints`."
    lon: Longitude in format 12E, 12.3E, 12.3, or 13W, 13.2W, -13.4. **Alternatively, you can specify the location by parameter `place_id`.**
        lat: Latitude in format 12N, 12.3N, 12.3, or 13S, 13.2S, -13.4. **Alternatively, you can specify the location by parameter `place_id`.**
        language: The language of text summaries (variable names are never translated). Available languages are:

- `en`: English
- `es`: Spanish
- `fr`: French
- `de`: German
- `pl`: Polish
- `cs`: Czech

        place_id: Identifier of a place. To obtain the `place_id` for the location you want, please use `Location endpoints`. **Alternatively, you can specify the location by parameters `lat` and `lon`.**
        units: Unit system to be used. The available values are:

- `auto`: Select the system automatically, based on the forecast location.
- `metric`: Metric (SI) units (`°C`, `mm/h`, `m/s`, `cm`, `km`, `hPa`).
- `us`: Imperial units (`°F`, `in/h`, `mph`, `in`, `mi`, `Hg`).
- `uk`: Same as `metric`, except that visibility is in `miles` and wind speeds are in `mph`.
- `ca`: Same as `metric`, except that wind speeds are in `km/h` and pressure is in `kPa`.

        timezone: Timezone to be used for the date fields. If not specified, local timezone of the forecast location will be used. The format is according to the tzinfo database, so values like `Europe/Prague` or `UTC` can be used. Alternatively you may use the value `auto` in which case the local timezone of the location is used. The full list of valid timezone strings can be found [here](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones#List).

        
    """
    url = f"https://ai-weather-by-meteosource.p.rapidapi.com/hourly"
    querystring = {}
    if lon:
        querystring['lon'] = lon
    if lat:
        querystring['lat'] = lat
    if language:
        querystring['language'] = language
    if place_id:
        querystring['place_id'] = place_id
    if units:
        querystring['units'] = units
    if timezone:
        querystring['timezone'] = timezone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ai-weather-by-meteosource.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def current(language: str='en', units: str='auto', place_id: str=None, lon: str='-122.42282', timezone: str='auto', lat: str='37.81021', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Current weather** conditions based on weather stations around the world.  Updated every 10 minutes. **Define your location** using GPS coordinates or `place_id` from `Location endpoints`."
    language: The language of text summaries (variable names are never translated). Available languages are:

- `en`: English
- `es`: Spanish
- `fr`: French
- `de`: German
- `pl`: Polish
- `cs`: Czech

        units: Unit system to be used. The available values are:

- `auto`: Select the system automatically, based on the forecast location.
- `metric`: Metric (SI) units (`°C`, `mm/h`, `m/s`, `cm`, `km`, `hPa`).
- `us`: Imperial units (`°F`, `in/h`, `mph`, `in`, `mi`, `Hg`).
- `uk`: Same as `metric`, except that visibility is in `miles` and wind speeds are in `mph`.
- `ca`: Same as `metric`, except that wind speeds are in `km/h` and pressure is in `kPa`.
        place_id: Identifier of a place. To obtain the `place_id` for the location you want, please use `Location endpoints`. **Alternatively, you can specify the location by parameters `lat` and `lon`.**
        lon: Longitude in format 12E, 12.3E, 12.3, or 13W, 13.2W, -13.4. **Alternatively, you can specify the location by parameter `place_id`.**
        timezone: Timezone to be used for the date fields. If not specified, local timezone of the forecast location will be used. The format is according to the tzinfo database, so values like `Europe/Prague` or `UTC` can be used. Alternatively you may use the value `auto` in which case the local timezone of the location is used. The full list of valid timezone strings can be found [here](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones#List).
        lat: Latitude in format 12N, 12.3N, 12.3, or 13S, 13.2S, -13.4. **Alternatively, you can specify the location by parameter `place_id`.**
        
    """
    url = f"https://ai-weather-by-meteosource.p.rapidapi.com/current"
    querystring = {}
    if language:
        querystring['language'] = language
    if units:
        querystring['units'] = units
    if place_id:
        querystring['place_id'] = place_id
    if lon:
        querystring['lon'] = lon
    if timezone:
        querystring['timezone'] = timezone
    if lat:
        querystring['lat'] = lat
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ai-weather-by-meteosource.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def daily(language: str='en', units: str='auto', lon: str='-122.42282', place_id: str=None, lat: str='37.81021', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Daily weather** forecast for the next 21 days. **Global** data are based on our **AI technology**, which uses many different models. **Define your location** using GPS coordinates or `place_id` from `Location endpoints`."
    language: The language of text summaries (variable names are never translated). Available languages are:

- `en`: English
- `es`: Spanish
- `fr`: French
- `de`: German
- `pl`: Polish
- `cs`: Czech

        units: Unit system to be used. The available values are:

- `auto`: Select the system automatically, based on the forecast location.
- `metric`: Metric (SI) units (`°C`, `mm/h`, `m/s`, `cm`, `km`, `hPa`).
- `us`: Imperial units (`°F`, `in/h`, `mph`, `in`, `mi`, `Hg`).
- `uk`: Same as `metric`, except that visibility is in `miles` and wind speeds are in `mph`.
- `ca`: Same as `metric`, except that wind speeds are in `km/h` and pressure is in `kPa`.

        lon: Longitude in format 12E, 12.3E, 12.3, or 13W, 13.2W, -13.4. **Alternatively, you can specify the location by parameter `place_id`.**
        place_id: Identifier of a place. To obtain the `place_id` for the location you want, please use `Location endpoints`. **Alternatively, you can specify the location by parameters `lat` and `lon`.**
        lat: Latitude in format 12N, 12.3N, 12.3, or 13S, 13.2S, -13.4. **Alternatively, you can specify the location by parameter `place_id`.**
        
    """
    url = f"https://ai-weather-by-meteosource.p.rapidapi.com/daily"
    querystring = {}
    if language:
        querystring['language'] = language
    if units:
        querystring['units'] = units
    if lon:
        querystring['lon'] = lon
    if place_id:
        querystring['place_id'] = place_id
    if lat:
        querystring['lat'] = lat
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ai-weather-by-meteosource.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def minutely(place_id: str=None, timezone: str='auto', lat: str='37.81021', language: str='en', units: str='auto', lon: str='-122.42282', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Minute-by-minute** precipitation forecast for the next 60 minutes. Updated in **real-time** based on our **AI nowcasting models**. **Define your location** using GPS coordinates or `place_id` from `Location endpoints`."
    place_id: Identifier of a place. To obtain the `place_id` for the location you want, please use `Location endpoints`. **Alternatively, you can specify the location by parameters `lat` and `lon`.**
        timezone: Timezone to be used for the date fields. If not specified, local timezone of the forecast location will be used. The format is according to the tzinfo database, so values like `Europe/Prague` or `UTC` can be used. Alternatively you may use the value `auto` in which case the local timezone of the location is used. The full list of valid timezone strings can be found [here](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones#List).

        lat: Latitude in format 12N, 12.3N, 12.3, or 13S, 13.2S, -13.4. **Alternatively, you can specify the location by parameter `place_id`.**
        language: The language of text summaries (variable names are never translated). Available languages are:

- `en`: English
- `es`: Spanish
- `fr`: French
- `de`: German
- `pl`: Polish
- `cs`: Czech

        units: Unit system to be used. The available values are:

- `auto`: Select the system automatically, based on the forecast location.
- `metric`: Metric (SI) units (`°C`, `mm/h`, `m/s`, `cm`, `km`, `hPa`).
- `us`: Imperial units (`°F`, `in/h`, `mph`, `in`, `mi`, `Hg`).
- `uk`: Same as `metric`, except that visibility is in `miles` and wind speeds are in `mph`.
- `ca`: Same as `metric`, except that wind speeds are in `km/h` and pressure is in `kPa`.

        lon: Longitude in format 12E, 12.3E, 12.3, or 13W, 13.2W, -13.4. **Alternatively, you can specify the location by parameter `place_id`.**
        
    """
    url = f"https://ai-weather-by-meteosource.p.rapidapi.com/minutely"
    querystring = {}
    if place_id:
        querystring['place_id'] = place_id
    if timezone:
        querystring['timezone'] = timezone
    if lat:
        querystring['lat'] = lat
    if language:
        querystring['language'] = language
    if units:
        querystring['units'] = units
    if lon:
        querystring['lon'] = lon
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ai-weather-by-meteosource.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def alerts(place_id: str=None, lat: str='45.74846', lon: str='4.84671', language: str='en', timezone: str='auto', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Severe weather alerts** for the USA, Europe, and Canada. **Define your location** using GPS coordinates or `place_id` from `Location` endpoints."
    place_id: Identifier of a place. To obtain the `place_id` for the location you want, please use `Location endpoints`. **Alternatively, you can specify the location by parameters `lat` and `lon`.**
        lat: Latitude in format 12N, 12.3N, 12.3, or 13S, 13.2S, -13.4. **Alternatively, you can specify the location by parameter `place_id`.**
        lon: Longitude in format 12E, 12.3E, 12.3, or 13W, 13.2W, -13.4. **Alternatively, you can specify the location by parameter `place_id`.**
        language: The language of text summaries (variable names are never translated). Available languages are:

- `en`: English
- `es`: Spanish
- `fr`: French
- `de`: German
- `pl`: Polish
- `cs`: Czech

        timezone: Timezone to be used for the date fields. If not specified, local timezone of the forecast location will be used. The format is according to the tzinfo database, so values like `Europe/Prague` or `UTC` can be used. Alternatively you may use the value `auto` in which case the local timezone of the location is used. The full list of valid timezone strings can be found [here](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones#List).

        
    """
    url = f"https://ai-weather-by-meteosource.p.rapidapi.com/alerts"
    querystring = {}
    if place_id:
        querystring['place_id'] = place_id
    if lat:
        querystring['lat'] = lat
    if lon:
        querystring['lon'] = lon
    if language:
        querystring['language'] = language
    if timezone:
        querystring['timezone'] = timezone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ai-weather-by-meteosource.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def find_places(text: str, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Search places by name** to get **place_id** for the `Weather Forecast Endpoints` and detailed **geographical information** (country, region, elevation, timezone, etc.) for a given location.
		
		The response can contain multiple places, sorted by relevance. You can then identify the one you want by coordinates, country, region, or type.
		
		Unlike the *find\_places\_prefix* endpoint, complete words are required here. You can search for cities, mountains, lakes, countries, etc."
    text: Place name to search for
        language: The language the place names. Available languages are:

- `en`: English
- `es`: Spanish
- `fr`: French
- `de`: German
- `pl`: Polish
- `cs`: Czech

        
    """
    url = f"https://ai-weather-by-meteosource.p.rapidapi.com/find_places"
    querystring = {'text': text, }
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ai-weather-by-meteosource.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def find_places_prefix(text: str, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Search places by beginning of the name** to get **place_id** for the `Weather Forecast Endpoints` and detailed **geographical information** (country, region, elevation, timezone, etc.) for a given location.
		
		The response can contain multiple places, sorted by relevance. You can then identify the one you want by coordinates, country, region, or type.
		
		Unlike the *find_places* endpoint, you can specify only the prefix of the place you are looking for. You can search for cities, mountains, lakes, countries, etc."
    text: Place name prefix to search for
        language: The language the place names. Available languages are:

- `en`: English
- `es`: Spanish
- `fr`: French
- `de`: German
- `pl`: Polish
- `cs`: Czech


        
    """
    url = f"https://ai-weather-by-meteosource.p.rapidapi.com/find_places_prefix"
    querystring = {'text': text, }
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ai-weather-by-meteosource.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nearest_place(lon: str, lat: str, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to search for the **nearest named place** (village/town/city) from a given GPS coordinates. You will get **place_id** for the `Weather Forecast Endpoints` and detailed **geographical information**.
		
		*Note: If you specify coordinates of a secluded place (e.g. middle of the ocean), the nearest point can be very far from the coordinates.*"
    lon: Longitude in format 12E, 12.3E, 12.3, or 13W, 13.2W, -13.4
        lat: Latitude in format 12N, 12.3N, 12.3, or 13S, 13.2S, -13.4
        language: The language the place names. Available languages are:

- `en`: English
- `es`: Spanish
- `fr`: French
- `de`: German
- `pl`: Polish
- `cs`: Czech


        
    """
    url = f"https://ai-weather-by-meteosource.p.rapidapi.com/nearest_place"
    querystring = {'lon': lon, 'lat': lat, }
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ai-weather-by-meteosource.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def astro(lon: str='-122.42282', place_id: str=None, lat: str='37.81021', timezone: str='auto', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns global **Sun and Moon information** (sunrise/sunset, moonrise/moonset and moon phase) for the next 30 days. Define your location using GPS coordinates or `place_id` from `Location endpoints`."
    lon: Longitude in format 12E, 12.3E, 12.3, or 13W, 13.2W, -13.4. **Alternatively, you can specify the location by parameter `place_id`.**
        place_id: Identifier of a place. To obtain the `place_id` for the location you want, please use `Location endpoints`. **Alternatively, you can specify the location by parameters `lat` and `lon`.**
        lat: Latitude in format 12N, 12.3N, 12.3, or 13S, 13.2S, -13.4. **Alternatively, you can specify the location by parameter `place_id`.**
        timezone: Timezone to be used for the date fields. If not specified, local timezone of the forecast location will be used. The format is according to the tzinfo database, so values like `Europe/Prague` or `UTC` can be used. Alternatively you may use the value `auto` in which case the local timezone of the location is used. The full list of valid timezone strings can be found [here](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones#List).

        
    """
    url = f"https://ai-weather-by-meteosource.p.rapidapi.com/astro"
    querystring = {}
    if lon:
        querystring['lon'] = lon
    if place_id:
        querystring['place_id'] = place_id
    if lat:
        querystring['lat'] = lat
    if timezone:
        querystring['timezone'] = timezone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ai-weather-by-meteosource.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

