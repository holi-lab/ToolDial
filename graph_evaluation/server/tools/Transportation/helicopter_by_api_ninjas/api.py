import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_helicopter(max_height: int=None, limit: int=None, min_height: int=None, max_length: int=None, min_length: int=None, min_speed: int=None, min_range: int=None, max_range: int=None, max_speed: int=None, manufacturer: str='Bell', model: str='206', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Ninjas Helicopter API endpoint. Returns a list of helicopter specs that match the given parameters.
		
		At least one of the following parameters (excluding the limit parameter) must be set."
    max_height: Maximum height of the helicopter in feet.
        limit: How many results to return. Must be between 1 and 30. Default is 1.
        min_height: Minimum height of the helicopter in feet.
        max_length: Maximum length of the helicopter in feet.
        min_length: Minimum length of the helicopter in feet.
        min_speed: Minimum max. air speed in knots.
        min_range: Minimum range of the helicopter in nautical miles.
        max_range: Maximum range of the helicopter in nautical miles.
        max_speed: Maximum max. air speed in knots.
        manufacturer: Company that designed and built the helicopter.
        model: Helicopter model name.
        
    """
    url = f"https://helicopter-by-api-ninjas.p.rapidapi.com/v1/helicopter"
    querystring = {}
    if max_height:
        querystring['max_height'] = max_height
    if limit:
        querystring['limit'] = limit
    if min_height:
        querystring['min_height'] = min_height
    if max_length:
        querystring['max_length'] = max_length
    if min_length:
        querystring['min_length'] = min_length
    if min_speed:
        querystring['min_speed'] = min_speed
    if min_range:
        querystring['min_range'] = min_range
    if max_range:
        querystring['max_range'] = max_range
    if max_speed:
        querystring['max_speed'] = max_speed
    if manufacturer:
        querystring['manufacturer'] = manufacturer
    if model:
        querystring['model'] = model
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "helicopter-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

