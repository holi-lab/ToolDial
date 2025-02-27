import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def route_and_directions(waypoints: str, mode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide Latitude/Longitude coordinates (separated by "|") as "waypoints", transportation mode ("drive", "truck", "bicycle", "walk" or "transit") as "mode". The API returns [GeoJSON.Features](https://apidocs.geoapify.com/docs/routing/feature/) as a result."
    waypoints: List of two or more lat,lon coordinates, separated by the pipe symbol \\\\\\\\\\\\\\\"|\\\\\\\\\\\\\\\"
        mode: Routing mode - drive, truck, bicycle, walk or transit
        
    """
    url = f"https://route-and-directions.p.rapidapi.com/v1/routing"
    querystring = {'waypoints': waypoints, 'mode': mode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "route-and-directions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

