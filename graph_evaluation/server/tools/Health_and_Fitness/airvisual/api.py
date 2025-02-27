import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def stations_v2_get_measurements(is_id: str, x_user_timezone: str='Asia/Singapore', x_units_temperature: str='celsius', x_user_lang: str='en-US', x_units_pressure: str='mbar', x_units_distance: str='kilometer', x_aqi_index: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get measurements at specific station by its id"
    id: The value of id field (type is station) that returned in …/v2/auto-complete endpoint
        x_units_temperature: One of the following : fahrenheit|celsius
        x_units_pressure: One of the following : hg|mbar
        x_units_distance: One of the following : miles|kilometer
        x_aqi_index: One of the following : us|cn
        
    """
    url = f"https://airvisual1.p.rapidapi.com/stations/v2/get-measurements"
    querystring = {'id': is_id, }
    if x_user_timezone:
        querystring['x-user-timezone'] = x_user_timezone
    if x_units_temperature:
        querystring['x-units-temperature'] = x_units_temperature
    if x_user_lang:
        querystring['x-user-lang'] = x_user_lang
    if x_units_pressure:
        querystring['x-units-pressure'] = x_units_pressure
    if x_units_distance:
        querystring['x-units-distance'] = x_units_distance
    if x_aqi_index:
        querystring['x-aqi-index'] = x_aqi_index
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airvisual1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cities_v2_get_measurements(is_id: str, x_units_distance: str='kilometer', x_units_pressure: str='mbar', x_aqi_index: str='us', x_units_temperature: str='celsius', x_user_timezone: str='Asia/Singapore', x_user_lang: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get measurements in specific city by its id"
    id: The value of id field (type is city) that returned in …/v2/auto-complete endpoint
        x_units_distance: One of the following : miles|kilometer
        x_units_pressure: One of the following : hg|mbar
        x_aqi_index: One of the following : us|cn
        x_units_temperature: One of the following : fahrenheit|celsius
        
    """
    url = f"https://airvisual1.p.rapidapi.com/cities/v2/get-measurements"
    querystring = {'id': is_id, }
    if x_units_distance:
        querystring['x-units-distance'] = x_units_distance
    if x_units_pressure:
        querystring['x-units-pressure'] = x_units_pressure
    if x_aqi_index:
        querystring['x-aqi-index'] = x_aqi_index
    if x_units_temperature:
        querystring['x-units-temperature'] = x_units_temperature
    if x_user_timezone:
        querystring['x-user-timezone'] = x_user_timezone
    if x_user_lang:
        querystring['x-user-lang'] = x_user_lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airvisual1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cities_v2_get_information(is_id: str, x_units_temperature: str='celsius', x_aqi_index: str='us', x_user_timezone: str='Asia/Singapore', x_user_lang: str='en-US', x_units_pressure: str='mbar', x_units_distance: str='kilometer', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information in specific city by its id"
    id: The value of id field (type is city) that returned in …/v2/auto-complete endpoint
        x_units_temperature: One of the following : fahrenheit|celsius
        x_aqi_index: One of the following : us|cn
        x_units_pressure: One of the following : hg|mbar
        x_units_distance: One of the following : miles|kilometer
        
    """
    url = f"https://airvisual1.p.rapidapi.com/cities/v2/get-information"
    querystring = {'id': is_id, }
    if x_units_temperature:
        querystring['x-units-temperature'] = x_units_temperature
    if x_aqi_index:
        querystring['x-aqi-index'] = x_aqi_index
    if x_user_timezone:
        querystring['x-user-timezone'] = x_user_timezone
    if x_user_lang:
        querystring['x-user-lang'] = x_user_lang
    if x_units_pressure:
        querystring['x-units-pressure'] = x_units_pressure
    if x_units_distance:
        querystring['x-units-distance'] = x_units_distance
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airvisual1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def places_v2_list_by_map(nelat: int, nelon: int, swlat: int, swlon: int, x_user_lang: str='en-US', x_units_pressure: str='mbar', x_units_temperature: str='celsius', x_user_timezone: str='Asia/Singapore', x_aqi_index: str='us', x_units_distance: str='kilometer', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List places in an area by providing a boundary box"
    nelat: North East latitude of boundary
        nelon: North East longitude of boundary
        swlat: Sount West latitude of boundary
        swlon: Sount West longitude of boundary
        x_units_pressure: One of the following : hg|mbar
        x_units_temperature: One of the following : fahrenheit|celsius
        x_aqi_index: One of the following : us|cn
        x_units_distance: One of the following : miles|kilometer
        
    """
    url = f"https://airvisual1.p.rapidapi.com/places/v2/list-by-map"
    querystring = {'NElat': nelat, 'NElon': nelon, 'SWlat': swlat, 'SWlon': swlon, }
    if x_user_lang:
        querystring['x-user-lang'] = x_user_lang
    if x_units_pressure:
        querystring['x-units-pressure'] = x_units_pressure
    if x_units_temperature:
        querystring['x-units-temperature'] = x_units_temperature
    if x_user_timezone:
        querystring['x-user-timezone'] = x_user_timezone
    if x_aqi_index:
        querystring['x-aqi-index'] = x_aqi_index
    if x_units_distance:
        querystring['x-units-distance'] = x_units_distance
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airvisual1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_auto_complete(q: str, x_units_pressure: str='mbar', x_aqi_index: str='us', x_units_temperature: str='celsius', x_units_distance: str='kilometer', x_user_timezone: str='Asia/Singapore', x_user_lang: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find countries, cities, places by name"
    q: Name of countries, cities, districts, places, etc...
        x_units_pressure: One of the following : hg|mbar
        x_aqi_index: One of the following : us|cn
        x_units_temperature: One of the following : fahrenheit|celsius
        x_units_distance: One of the following : miles|kilometer
        
    """
    url = f"https://airvisual1.p.rapidapi.com/v2/auto-complete"
    querystring = {'q': q, }
    if x_units_pressure:
        querystring['x-units-pressure'] = x_units_pressure
    if x_aqi_index:
        querystring['x-aqi-index'] = x_aqi_index
    if x_units_temperature:
        querystring['x-units-temperature'] = x_units_temperature
    if x_units_distance:
        querystring['x-units-distance'] = x_units_distance
    if x_user_timezone:
        querystring['x-user-timezone'] = x_user_timezone
    if x_user_lang:
        querystring['x-user-lang'] = x_user_lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airvisual1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stations_get_information_deprecated(is_id: str, aqiindex: str='us', timezone: str='Asia/Singapore', lang: str='en_US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information at specific station by its id"
    id: The value of id field (type is station) that received from .../auto-complete API
        aqiindex: AQI index (us | cn)
        
    """
    url = f"https://airvisual1.p.rapidapi.com/stations/get-information"
    querystring = {'id': is_id, }
    if aqiindex:
        querystring['aqiIndex'] = aqiindex
    if timezone:
        querystring['timezone'] = timezone
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airvisual1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stations_get_measurements_deprecated(is_id: str, lang: str='en_US', aqiindex: str='us', timezone: str='Asia/Singapore', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get measurements at specific station by its id"
    id: The value of id field (type is station) that received from .../auto-complete API
        aqiindex: AQI index (us | cn)
        
    """
    url = f"https://airvisual1.p.rapidapi.com/stations/get-measurements"
    querystring = {'id': is_id, }
    if lang:
        querystring['lang'] = lang
    if aqiindex:
        querystring['aqiIndex'] = aqiindex
    if timezone:
        querystring['timezone'] = timezone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airvisual1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cities_get_information_deprecated(is_id: str, aqiindex: str='us', timezone: str='Asia/Singapore', lang: str='en_US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information in specific city by its id"
    id: The value of id field (type is city) that received from .../auto-complete API
        aqiindex: AQI index (us | cn)
        
    """
    url = f"https://airvisual1.p.rapidapi.com/cities/get-information"
    querystring = {'id': is_id, }
    if aqiindex:
        querystring['aqiIndex'] = aqiindex
    if timezone:
        querystring['timezone'] = timezone
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airvisual1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cities_get_measurements_deprecated(is_id: str, timezone: str='Asia/Singapore', lang: str='en_US', aqiindex: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get measurements in specific city by its id"
    id: The value of id field (type is city) that received from .../auto-complete API
        
    """
    url = f"https://airvisual1.p.rapidapi.com/cities/get-measurements"
    querystring = {'id': is_id, }
    if timezone:
        querystring['timezone'] = timezone
    if lang:
        querystring['lang'] = lang
    if aqiindex:
        querystring['aqiIndex'] = aqiindex
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airvisual1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stations_v2_get_information(is_id: str, x_user_timezone: str='Asia/Singapore', x_units_distance: str='kilometer', x_units_pressure: str='mbar', x_user_lang: str='en-US', x_units_temperature: str='celsius', x_aqi_index: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information at specific station by its id"
    id: The value of id field (type is station) that returned in …/v2/auto-complete endpoint
        x_units_distance: One of the following : miles|kilometer
        x_units_pressure: One of the following : hg|mbar
        x_units_temperature: One of the following : fahrenheit|celsius
        x_aqi_index: One of the following : us|cn
        
    """
    url = f"https://airvisual1.p.rapidapi.com/stations/v2/get-information"
    querystring = {'id': is_id, }
    if x_user_timezone:
        querystring['x-user-timezone'] = x_user_timezone
    if x_units_distance:
        querystring['x-units-distance'] = x_units_distance
    if x_units_pressure:
        querystring['x-units-pressure'] = x_units_pressure
    if x_user_lang:
        querystring['x-user-lang'] = x_user_lang
    if x_units_temperature:
        querystring['x-units-temperature'] = x_units_temperature
    if x_aqi_index:
        querystring['x-aqi-index'] = x_aqi_index
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airvisual1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def places_list_by_map_deprecated(nelon: int, swlat: int, nelat: int, swlon: int, maptype: str, zoomlevel: int, lang: str='en_US', aqiindex: str='us', timezone: str='Asia/Singapore', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List places in an area by providing a boundary box"
    nelon: North East longitude of boundary
        swlat: Sount West latitude of boundary
        nelat: North East latitude of boundary
        swlon: Sount West longitude of boundary
        maptype: GoogleMap or China is allowed
        zoomlevel: Zoom level of map, this value affects how many places returned in response
        lang: Language code
        aqiindex: AQI index (us | cn)
        timezone: Timezone
        
    """
    url = f"https://airvisual1.p.rapidapi.com/places/list-by-map"
    querystring = {'NElon': nelon, 'SWlat': swlat, 'NElat': nelat, 'SWlon': swlon, 'mapType': maptype, 'zoomLevel': zoomlevel, }
    if lang:
        querystring['lang'] = lang
    if aqiindex:
        querystring['aqiIndex'] = aqiindex
    if timezone:
        querystring['timezone'] = timezone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airvisual1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def auto_complete_deprecated(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find countries, cities, places by name"
    query: Name of countries, cities, districts, places, etc...
        
    """
    url = f"https://airvisual1.p.rapidapi.com/auto-complete"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airvisual1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

