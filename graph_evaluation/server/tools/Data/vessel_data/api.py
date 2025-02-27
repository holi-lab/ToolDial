import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_vessel_photo(shipid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Image of vessel based on given ship id.
		*Image might not be display correctly here due to base64 issue of Test Endpoint, but it work indeed.*"
    
    """
    url = f"https://vessel-data.p.rapidapi.com/get_vessel_photo/{shipid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vessel-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_vessels(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return all available on board vessels.
		Visit our [demo site](https://core-api.net/vessel/map.html)"
    
    """
    url = f"https://vessel-data.p.rapidapi.com/get_all_vessels"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vessel-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_vessel_info(shipid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Individual on board vessel info based on given ship id.
		Visit our [demo site](https://core-api.net/vessel/map.html)"
    shipid: Ship ID which included in list return by **Get All Vessels** or **Get Vessels by Area** method.
e.g "SHIP_ID": "410553"
        
    """
    url = f"https://vessel-data.p.rapidapi.com/get_vessel_info/{shipid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vessel-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_vessels_by_geo_position(latitude: int, longitude: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return available vessels within 300 nautical miles radius of the given geo coordinate.
		Visit our [demo site](https://core-api.net/vessel/map.html)"
    
    """
    url = f"https://vessel-data.p.rapidapi.com/get_vessels_by_position"
    querystring = {'latitude': latitude, 'longitude': longitude, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vessel-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_vessels_by_ship_name(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return vessels by its name.
		Visit our [demo site](https://core-api.net/vessel/map.html)"
    name: Name of the ship
        
    """
    url = f"https://vessel-data.p.rapidapi.com/get_vessels_by_name"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vessel-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

