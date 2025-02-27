import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_by_geohash(hash: str, count: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search all the Gas Stations and Fuel Prices by providing a gehash instead of Lat/Lng/Radius
		For info: [Wikipedia - Geohash](https://en.wikipedia.org/wiki/Geohash)"
    
    """
    url = f"https://fuel-prices-gas-stations-italy.p.rapidapi.com/search/geohash"
    querystring = {'hash': hash, }
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fuel-prices-gas-stations-italy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_by_latitude_longitude(lng: int, radius: int, lat: int, count: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Define a location with Latitude and Longitude and a Radius to search within. You can limit the maximum numbers of results."
    radius: Radius defined in Meters (m)
        
    """
    url = f"https://fuel-prices-gas-stations-italy.p.rapidapi.com/search/location"
    querystring = {'lng': lng, 'radius': radius, 'lat': lat, }
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fuel-prices-gas-stations-italy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

