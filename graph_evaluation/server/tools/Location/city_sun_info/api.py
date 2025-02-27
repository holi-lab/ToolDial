import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def sun_info(city: str, date: str, state: str='Massachusetts', country: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the sun information for a city on a given date, such as sunrise, sunset, length of day, and other fields. 
		The time will be converted to the timezone of the provided city."
    
    """
    url = f"https://city-sun-info.p.rapidapi.com/city"
    querystring = {'city': city, 'date': date, }
    if state:
        querystring['state'] = state
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "city-sun-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

