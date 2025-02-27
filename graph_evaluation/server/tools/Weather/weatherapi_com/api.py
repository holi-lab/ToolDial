import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def realtime_weather_api(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Current weather or realtime weather API method allows a user to get up to date current weather information in json and xml. The data is returned as a Current Object."
    q: Query parameter based on which data is sent back. It could be following:  Latitude and Longitude (Decimal degree) e.g: q=48.8567,2.3508 city name e.g.: q=Paris US zip e.g.: q=10001 UK postcode e.g: q=SW1 Canada postal code e.g: q=G2J metar:<metar code> e.g: q=metar:EGLL iata:<3 digit airport code> e.g: q=iata:DXB auto:ip IP lookup e.g: q=auto:ip IP address (IPv4 and IPv6 supported) e.g: q=100.0.0.1
        
    """
    url = f"https://weatherapi-com.p.rapidapi.com/current.json"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weatherapi-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def forecast_weather_api(q: str, days: int=3, dt: str=None, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Forecast weather API method returns upto next 14 day weather forecast and weather alert as json. It contains astronomy data, day weather forecast and hourly interval weather information for a given city."
    q: Query parameter based on which data is sent back. It could be following:

Latitude and Longitude (Decimal degree) e.g: q=48.8567,2.3508
city name e.g.: q=Paris
US zip e.g.: q=10001
UK postcode e.g: q=SW1
Canada postal code e.g: q=G2J
metar:<metar code> e.g: q=metar:EGLL
iata:<3 digit airport code> e.g: q=iata:DXB
auto:ip IP lookup e.g: q=auto:ip
IP address (IPv4 and IPv6 supported) e.g: q=100.0.0.1
        days: Number of days of forecast required.
        dt: If passing 'dt', it should be between today and next 10 day in yyyy-MM-dd format.
        lang: Returns 'condition:text' field in API in the desired language
        
    """
    url = f"https://weatherapi-com.p.rapidapi.com/forecast.json"
    querystring = {'q': q, }
    if days:
        querystring['days'] = days
    if dt:
        querystring['dt'] = dt
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weatherapi-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def time_zone_api(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Time Zone API method allows a user to get up to date time zone and local time information in json."
    q: Query parameter based on which data is sent back. It could be following:

Latitude and Longitude (Decimal degree) e.g: q=48.8567,2.3508
city name e.g.: q=Paris
US zip e.g.: q=10001
UK postcode e.g: q=SW1
Canada postal code e.g: q=G2J
metar:<metar code> e.g: q=metar:EGLL
iata:<3 digit airport code> e.g: q=iata:DXB
auto:ip IP lookup e.g: q=auto:ip
IP address (IPv4 and IPv6 supported) e.g: q=100.0.0.1
        
    """
    url = f"https://weatherapi-com.p.rapidapi.com/timezone.json"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weatherapi-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def future_weather_api(dt: str, q: str, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Future weather API method returns weather in a 3 hourly interval in future for a date between 14 days and 300 days from today in the future."
    dt: 'dt' should be between 14 days and 300 days from today in the future in yyyy-MM-dd format (i.e. dt=2023-01-01)
        q: Query parameter based on which data is sent back. It could be following:

Latitude and Longitude (Decimal degree) e.g: q=48.8567,2.3508
city name e.g.: q=Paris
US zip e.g.: q=10001
UK postcode e.g: q=SW1
Canada postal code e.g: q=G2J
metar:<metar code> e.g: q=metar:EGLL
iata:<3 digit airport code> e.g: q=iata:DXB
auto:ip IP lookup e.g: q=auto:ip
IP address (IPv4 and IPv6 supported) e.g: q=100.0.0.1
        lang: Returns 'condition:text' field in API in the desired language
        
    """
    url = f"https://weatherapi-com.p.rapidapi.com/future.json"
    querystring = {'dt': dt, 'q': q, }
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weatherapi-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sports_api(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Sports API method allows a user to get listing of all upcoming sports events for football, cricket and golf in json."
    
    """
    url = f"https://weatherapi-com.p.rapidapi.com/sports.json"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weatherapi-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def history_weather_api(dt: str, q: str, hour: int=None, end_dt: str=None, lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "History weather API method returns historical weather for a date on or after 1st Jan, 2010 (depending upon subscription level) as json."
    dt: For history API 'dt' should be on or after 1st Jan, 2010 in yyyy-MM-dd format
        q: Query parameter based on which data is sent back. It could be following:

Latitude and Longitude (Decimal degree) e.g: q=48.8567,2.3508
city name e.g.: q=Paris
US zip e.g.: q=10001
UK postcode e.g: q=SW1
Canada postal code e.g: q=G2J
metar:<metar code> e.g: q=metar:EGLL
iata:<3 digit airport code> e.g: q=iata:DXB
auto:ip IP lookup e.g: q=auto:ip
IP address (IPv4 and IPv6 supported) e.g: q=100.0.0.1
        hour: Restricting history output to a specific hour in a given day.
        end_dt: Restrict date output for History API method. Should be on or after 1st Jan, 2010. Make sure end_dt is equal to or greater than 'dt'. 
        lang: Returns 'condition:text' field in API in the desired language
        
    """
    url = f"https://weatherapi-com.p.rapidapi.com/history.json"
    querystring = {'dt': dt, 'q': q, }
    if hour:
        querystring['hour'] = hour
    if end_dt:
        querystring['end_dt'] = end_dt
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weatherapi-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ip_lookup_api(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "IP Lookup API method allows a user to get up to date information for an IP address in json."
    q: e.g: q=auto:ip

IP address (IPv4 and IPv6 supported) e.g: q=100.0.0.1
        
    """
    url = f"https://weatherapi-com.p.rapidapi.com/ip.json"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weatherapi-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def astronomy_api(q: str, dt: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Astronomy API method allows a user to get up to date information for sunrise, sunset, moonrise, moonset, moon phase and illumination in json."
    q: Query parameter based on which data is sent back. It could be following:

Latitude and Longitude (Decimal degree) e.g: q=48.8567,2.3508
city name e.g.: q=Paris
US zip e.g.: q=10001
UK postcode e.g: q=SW1
Canada postal code e.g: q=G2J
metar:<metar code> e.g: q=metar:EGLL
iata:<3 digit airport code> e.g: q=iata:DXB
auto:ip IP lookup e.g: q=auto:ip
IP address (IPv4 and IPv6 supported) e.g: q=100.0.0.1
        dt: Date
        
    """
    url = f"https://weatherapi-com.p.rapidapi.com/astronomy.json"
    querystring = {'q': q, }
    if dt:
        querystring['dt'] = dt
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weatherapi-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_autocomplete_api(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search or Autocomplete API returns matching cities and towns."
    q: Query parameter based on which data is sent back. It could be following:

Latitude and Longitude (Decimal degree) e.g: q=48.8567,2.3508
city name e.g.: q=Paris
US zip e.g.: q=10001
UK postcode e.g: q=SW1
Canada postal code e.g: q=G2J
metar:<metar code> e.g: q=metar:EGLL
iata:<3 digit airport code> e.g: q=iata:DXB
auto:ip IP lookup e.g: q=auto:ip
IP address (IPv4 and IPv6 supported) e.g: q=100.0.0.1
        
    """
    url = f"https://weatherapi-com.p.rapidapi.com/search.json"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weatherapi-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

