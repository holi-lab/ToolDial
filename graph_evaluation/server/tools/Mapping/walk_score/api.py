import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def walk_score(lat: str, address: str, wsapikey: str, lon: str, format: str=None, bike: str=None, transit: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Walk Score"
    lat: 	The latitude of the requested location.
        address: The URL encoded address.
        wsapikey: Your Walk Score API Key. https://www.walkscore.com/professional/api-sign-up.php
        lon: 	The longitude of the requested location.
        format: Type of result to return: (movie, series, episode)
        bike: 	Set bike=1 to request Bike Score (if available).
        transit: Set transit=1 to request Transit Score (if available).
        
    """
    url = f"https://walk-score.p.rapidapi.com/score"
    querystring = {'lat': lat, 'address': address, 'wsapikey': wsapikey, 'lon': lon, }
    if format:
        querystring['format'] = format
    if bike:
        querystring['bike'] = bike
    if transit:
        querystring['transit'] = transit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "walk-score.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

