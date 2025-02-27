import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_distance(route: str, car: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a distance between locations"
    route: A route is described of an array of point objects. Every point object exists of a \"t\" property, which describes the name of the point. (Can be address, city, iata or lat,lng ) and an optional \"c\" propertie which describes the country in ISO 3166-1 alpha-2 or alpha-3 (DE or DEU, ES, US,...) Example: [{\"t\":\"TXL\"},{\"t\":\"Hamburg\"}]
        car: Add car route distance and duration
        
    """
    url = f"https://distanceto.p.rapidapi.com/get"
    querystring = {'route': route, }
    if car:
        querystring['car'] = car
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "distanceto.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

