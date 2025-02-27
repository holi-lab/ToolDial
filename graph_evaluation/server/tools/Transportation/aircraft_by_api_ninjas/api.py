import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_aircraft(min_height: int=None, min_wingspan: int=None, limit: int=None, max_height: int=None, max_length: int=None, min_length: int=None, max_range: int=None, min_range: int=None, max_speed: int=None, max_wingspan: int=None, engine_type: str=None, min_speed: int=None, model: str='G550', manufacturer: str='Gulfstream', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Ninjas Aircraft API endpoint. Returns a list of aircrafts that match the given parameters. This API only supports airplanes - for helicopter specs please use our Helicopter API.
		
		At least one of the following parameters (excluding the limit parameter) must be set."
    min_height: Minimum height of the aircraft in feet.
        min_wingspan: Minimum wingspan of the aircraft in feet.
        limit: How many results to return. Must be between 1 and 30. Default is 1.
        max_height: Maximum height of the aircraft in feet.
        max_length: Maximum length of the aircraft in feet.
        min_length: Minimum length of the aircraft in feet.
        max_range: Maximum range of the aircraft in nautical miles.
        min_range: Minimum range of the aircraft in nautical miles.
        max_speed: Maximum max. air speed in knots.
        max_wingspan: Maximum wingspan of the aircraft in feet.
        engine_type: Type of engine. Must be one of: piston, propjet, jet.
        min_speed: Minimum max. air speed in knots.
        model: Aircraft model name.
        manufacturer: Company that designed and built the aircraft.
        
    """
    url = f"https://aircraft-by-api-ninjas.p.rapidapi.com/v1/aircraft"
    querystring = {}
    if min_height:
        querystring['min_height'] = min_height
    if min_wingspan:
        querystring['min_wingspan'] = min_wingspan
    if limit:
        querystring['limit'] = limit
    if max_height:
        querystring['max_height'] = max_height
    if max_length:
        querystring['max_length'] = max_length
    if min_length:
        querystring['min_length'] = min_length
    if max_range:
        querystring['max_range'] = max_range
    if min_range:
        querystring['min_range'] = min_range
    if max_speed:
        querystring['max_speed'] = max_speed
    if max_wingspan:
        querystring['max_wingspan'] = max_wingspan
    if engine_type:
        querystring['engine_type'] = engine_type
    if min_speed:
        querystring['min_speed'] = min_speed
    if model:
        querystring['model'] = model
    if manufacturer:
        querystring['manufacturer'] = manufacturer
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aircraft-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

