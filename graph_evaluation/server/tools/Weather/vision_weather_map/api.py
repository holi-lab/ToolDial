import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def view_from_16_day_daily_forecast_dates(units: str='imperial', mode: str=None, lang: str=None, q: str='dallas,us', lon: int=138, is_id: str=None, cnt: int=10, zip: str=None, lat: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "16-day forecasts are available in any location or city. Forecasts include the daily weather and are available in JSON or XML format. It is only available for all paid accounts."
    units: You can use different types of metric systems by units = metric or imperial
        mode: 
mode - possible values are JSON xml. If mode parameter is empty the format is JSON by default.
        lang: You can use the lang parameter to get the output in your language. We support the following languages, which you can use with the corresponding lang values: Arabic - ar, Bulgarian - bg, Catalan - ca, Czech - cz, German - de, Greek - el, English - en, Persian (Farsi) - fa, Finnish - fi, French - fr, Galician - gl, Croatian - hr, Hungarian - hu, Italian - it, Japanese - ja, Korean - kr, Latvian - la, Lithuanian - lt, Macedonian - mk, Dutch - nl, Polish - pl, Portuguese - pt, Romanian - ro, Russian - ru, Swedish - se, Slovak - sk, Slovenian - sl, Spanish - es, Turkish - tr, Ukrainian - ua, Vietnamese - vi, Simplified Chinese - zhcn, Traditional Chinese - zhtw.
        q: city name and country code divided by comma, use ISO 3166 country codes
        lon: Must be used with lat. Get current weather data when you know the longitude of the city.
        is_id: city id


        cnt: 
amount of days in the future to forecast
        zip: {zip code},{country code}
        
    """
    url = f"https://vision-weather-map.p.rapidapi.com/16-day/"
    querystring = {}
    if units:
        querystring['units'] = units
    if mode:
        querystring['mode'] = mode
    if lang:
        querystring['lang'] = lang
    if q:
        querystring['q'] = q
    if lon:
        querystring['lon'] = lon
    if is_id:
        querystring['id'] = is_id
    if cnt:
        querystring['cnt'] = cnt
    if zip:
        querystring['zip'] = zip
    if lat:
        querystring['lat'] = lat
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vision-weather-map.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def climate_forecast_for_30_days(q: str='dallas,us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With the climate forecast for 30 days you can request weather data for the next 30 days. This product is based on a statistical approach for our historical weather data and is updated every hour."
    
    """
    url = f"https://vision-weather-map.p.rapidapi.com/30-days/"
    querystring = {}
    if q:
        querystring['q'] = q
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vision-weather-map.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_5_day_3_hour_forecast_data(q: str='dallas,us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The 5-day forecast is available in any location or city. It contains weather data every 3 hours. The forecast is available in JSON or XML format."
    q: {city name}, {country code} - city name and country code divided by comma, use ISO 3166 country codes
        
    """
    url = f"https://vision-weather-map.p.rapidapi.com/5-3-day/"
    querystring = {}
    if q:
        querystring['q'] = q
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vision-weather-map.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def current_weather_data(q: str, lang: str='null', mode: str='xml', units: str='imperial', lon: int=0, lat: int=0, is_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With this type of request you can get weather data anywhere on earth. The current weather data is updated online based on data from more than 40,000 weather stations."
    q: 
use this parameter when searching for a city. Do not use with other parameters
        lang: You can use the lang parameter to get the output in your language. We support the following languages, which you can use with the corresponding lang values: English - en, Russian - ru, Italian - it, Spanish - sp, Ukrainian - ua, German - de, Portuguese - pt, Romanian - ro, Polish - pl, Finnish - fi, Dutch - nl, French - fr, Bulgarian - bg, Swedish - se, Traditional Chinese - zht, Simplified Chinese - zhcn, Turkish - tr
        mode: If left blank will default to JSON output. Ability to retrieve data in XML or HTML.
        units: You can use different types of metric systems by units = metric or imperial
        lon: Must be used with lat. Get current weather data when you know the longitude of the city.
        lat: Must be used with lon. Get current weather data when you know the latitude of the city.
        is_id: Get current weather data when you know the city ID. Not to be used with lon, lat, or q
        
    """
    url = f"https://vision-weather-map.p.rapidapi.com/Current-weather/"
    querystring = {'q': q, }
    if lang:
        querystring['lang'] = lang
    if mode:
        querystring['mode'] = mode
    if units:
        querystring['units'] = units
    if lon:
        querystring['lon'] = lon
    if lat:
        querystring['lat'] = lat
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vision-weather-map.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

