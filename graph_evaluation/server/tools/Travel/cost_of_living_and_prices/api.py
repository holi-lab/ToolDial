import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def cities(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of all the cities in the database with corresponding country name, id, lat and lng"
    
    """
    url = f"https://cost-of-living-and-prices.p.rapidapi.com/cities"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cost-of-living-and-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def prices(country_name: str, city_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Prices for goods in services for city.  Require to specify city_name and country_name. Tip: See cities endpoint for list of cities and corresponding id, country_name, lat and lng."
    
    """
    url = f"https://cost-of-living-and-prices.p.rapidapi.com/prices"
    querystring = {'country_name': country_name, 'city_name': city_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cost-of-living-and-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

