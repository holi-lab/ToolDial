import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_stores_in_coords_range(long: str, lat: str, km: str=None, miles: str='3', meters: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve stores based off a coordinate query, this must include a lat, long, and range specified by miles, meters, or km."
    
    """
    url = f"https://speedway-gas-prices.p.rapidapi.com/getStoresCoords"
    querystring = {'long': long, 'lat': lat, }
    if km:
        querystring['km'] = km
    if miles:
        querystring['miles'] = miles
    if meters:
        querystring['meters'] = meters
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "speedway-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_stores_from_address(zip: str='45331', state: str='OH', city: str='Greenville', street: str='Sweitzer Street', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve stores based off a location query, this could include a street, city, state, and/or zip."
    
    """
    url = f"https://speedway-gas-prices.p.rapidapi.com/getStoresAddress"
    querystring = {}
    if zip:
        querystring['zip'] = zip
    if state:
        querystring['state'] = state
    if city:
        querystring['city'] = city
    if street:
        querystring['street'] = street
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "speedway-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

