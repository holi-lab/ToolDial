import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def calculatedrivingdistancematrix(origins: str, destinations: str='40.629041,-74.025606;40.630099,-73.993521;40.644895,-74.013818;40.627177,-73.980853', start_time: str=None, avoid_tolls: bool=None, avoid_highways: bool=None, avoid_ferries: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculate distances and durations between a set of origins and destinations."
    origins: List of origins described as semicolon-delimited coordinate pairs with latitudes and longitudes. Max: 25
        destinations: List of destinations described as semicolon-delimited coordinate pairs with latitudes and longitudes. If not specified, an n x n matrix will be generated using the origins. Max: 25
        start_time: Time when travel is expected to start. You can specify the time as an integer in seconds since midnight, January 1, 1970 UTC or you can use \"now\" to specify the current time.
        avoid_tolls: avoid toll roads
        avoid_highways: avoid highways
        avoid_ferries: avoid ferries
        
    """
    url = f"https://trueway-matrix.p.rapidapi.com/CalculateDrivingMatrix"
    querystring = {'origins': origins, }
    if destinations:
        querystring['destinations'] = destinations
    if start_time:
        querystring['start_time'] = start_time
    if avoid_tolls:
        querystring['avoid_tolls'] = avoid_tolls
    if avoid_highways:
        querystring['avoid_highways'] = avoid_highways
    if avoid_ferries:
        querystring['avoid_ferries'] = avoid_ferries
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trueway-matrix.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

