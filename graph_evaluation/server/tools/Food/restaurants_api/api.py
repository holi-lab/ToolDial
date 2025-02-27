import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_restaurants(latitude: int, radius: int, longitude: int, rangeprice: int=None, attendees: str=None, nextpagetoken: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "let you search for restaurants within a specified area."
    
    """
    url = f"https://restaurants-api.p.rapidapi.com/restaurants"
    querystring = {'latitude': latitude, 'radius': radius, 'longitude': longitude, }
    if rangeprice:
        querystring['rangePrice'] = rangeprice
    if attendees:
        querystring['attendees'] = attendees
    if nextpagetoken:
        querystring['nextPageToken'] = nextpagetoken
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "restaurants-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

