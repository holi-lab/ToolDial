import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_by_filter(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get by filter"
    
    """
    url = f"https://world-population-by-decade-and-growth-rate.p.rapidapi.com/yDRaVL/world_population_by_decade?Year=value"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-population-by-decade-and-growth-rate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_by_decade(decade: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get by decade"
    
    """
    url = f"https://world-population-by-decade-and-growth-rate.p.rapidapi.com/yDRaVL/world_population_by_decade"
    querystring = {}
    if decade:
        querystring['decade'] = decade
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-population-by-decade-and-growth-rate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

