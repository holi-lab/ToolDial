import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def gas_price(radius: int=10, fuel: str='E10', lon: int=10, lat: int=48, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Current gas price for a location"
    
    """
    url = f"https://gas-price-germany.p.rapidapi.com/benzinpreis.preis.json"
    querystring = {}
    if radius:
        querystring['radius'] = radius
    if fuel:
        querystring['fuel'] = fuel
    if lon:
        querystring['lon'] = lon
    if lat:
        querystring['lat'] = lat
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gas-price-germany.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ping(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Test the API Endpoint"
    
    """
    url = f"https://gas-price-germany.p.rapidapi.com/core.ping.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gas-price-germany.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

