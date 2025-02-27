import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_tide_extreme_data(lng: str, apikey: str, lat: str, enddate: str=None, startdate: str=None, datum: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Stormglass Tide Extreme Data**
		
		Retrieve information about high and low tide for a single coordinate. If nothing is specified, the returned values will be in relative to Mean Sea Level - MSL."
    
    """
    url = f"https://stormglass-complete.p.rapidapi.com/tideex/{apikey}/{lat}/{lng}/{startDate}/{endDate}/{datum}"
    querystring = {}
    if enddate:
        querystring['endDate'] = enddate
    if startdate:
        querystring['startDate'] = startdate
    if datum:
        querystring['datum'] = datum
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stormglass-complete.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_tide_sea_level_data(params: str, apikey: str, lat: str, lng: str, datum: str=None, startdate: str=None, enddate: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Stormglass Tide Sea-level Data**
		
		Retrieve the sea level given in meters hour by hour for a single coordinate. If nothing is specified the returned values will be in relative to Mean Sea Level - MSL."
    
    """
    url = f"https://stormglass-complete.p.rapidapi.com/tidesl/{apikey}/{lat}/{lng}/{params}/{startDate}/{endDate}/{datum}"
    querystring = {}
    if datum:
        querystring['datum'] = datum
    if startdate:
        querystring['startDate'] = startdate
    if enddate:
        querystring['endDate'] = enddate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stormglass-complete.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_astronomy_data(lat: str, lng: str, startdate: str=None, enddate: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Stormglass Astronomy Data
		
		Retrieve sunrise, sunset, moonrise, moonset and moon phase for a single coordinate."
    
    """
    url = f"https://stormglass-complete.p.rapidapi.com/astro/{lat}/{lng}/{startDate}/{endDate}"
    querystring = {}
    if startdate:
        querystring['startDate'] = startdate
    if enddate:
        querystring['endDate'] = enddate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stormglass-complete.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_tide_station_area_data(botlng: str, apikey: str, toplng: str, toplat: str, botlat: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Stormglass Tide Station Area Data**
		
		The Tide Stations Area Request will list all available tide stations within a defined geographic area.
		
		Top right and bottom left coordinate draws a box with the selected coordinates. 
		All stations with in the area of this box are queried."
    
    """
    url = f"https://stormglass-complete.p.rapidapi.com/tidesar/{apikey}/{toplat}/{toplng}/{botlat}/{botlng}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stormglass-complete.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

