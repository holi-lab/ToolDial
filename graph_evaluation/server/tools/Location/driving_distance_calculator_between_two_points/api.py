import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def origin_and_destination_location(destination: str, origin: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "It accept origin and destination city, place name, address or zip code."
    
    """
    url = f"https://driving-distance-calculator-between-two-points.p.rapidapi.com/data"
    querystring = {'destination': destination, 'origin': origin, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "driving-distance-calculator-between-two-points.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

