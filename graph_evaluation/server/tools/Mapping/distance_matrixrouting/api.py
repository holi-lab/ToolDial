import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def distance_calculation(destination: str, origin: str, priority: str='fast', vehicle: str='auto', units: str='ml', order: str='lat_lon', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows to get travel distance and time for a matrix of origin and destination. Response of this API looks like Google Maps API compact response."
    destination: One location to use as the final destination for calculating travel distance and time. Location in the form of latitude/longitude coordinates or longitude/latitude coordinates
        origin: Starting point for calculating travel distance and time. Location in the form of latitude/longitude coordinates or longitude/latitude coordinates
        priority: You may specify a priority of your route. It can be the fastest route or the shortest route.
        vehicle: For the calculation of distances, you may specify the transportation mode to use. By default, distances are calculated for the driving mode. The following travel modes are supported: auto, walk, truck
        units: You can specify units for distance. It can be miles or kilometers. Default units are miles. The following units are supported: ml, km
        order: You may specify an order of what stands first in origin and destination. lat_lon stands for Latitude/Longitude. lon_lat stands for Longitude/Latitude
        
    """
    url = f"https://distance-matrix-routing.p.rapidapi.com/distance"
    querystring = {'destination': destination, 'origin': origin, }
    if priority:
        querystring['priority'] = priority
    if vehicle:
        querystring['vehicle'] = vehicle
    if units:
        querystring['units'] = units
    if order:
        querystring['order'] = order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "distance-matrix-routing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

