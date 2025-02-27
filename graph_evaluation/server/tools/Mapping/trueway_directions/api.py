import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def finddrivingroute(stops: str, avoid_tolls: bool=None, geometry_format: str=None, optimize: bool=None, avoid_ferries: bool=None, avoid_highways: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find the best route and get multiple stops driving directions"
    stops: Specify two or more semicolon-delimited locations(Lat/Lng) between which the route is to be found. Max: 25
        avoid_tolls: the route should avoid toll roads
        geometry_format: geometry format: latlng, lnglat or geojson
        optimize: Instructs the API to reorder stops to find the optimized route. The route first stop and last stop order is not changed, their position is considered fixed.
        avoid_ferries: the route should avoid ferries
        avoid_highways: the route should avoid highways
        
    """
    url = f"https://trueway-directions2.p.rapidapi.com/FindDrivingRoute"
    querystring = {'stops': stops, }
    if avoid_tolls:
        querystring['avoid_tolls'] = avoid_tolls
    if geometry_format:
        querystring['geometry_format'] = geometry_format
    if optimize:
        querystring['optimize'] = optimize
    if avoid_ferries:
        querystring['avoid_ferries'] = avoid_ferries
    if avoid_highways:
        querystring['avoid_highways'] = avoid_highways
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trueway-directions2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def finddrivingpath(destination: str, origin: str, geometry_format: str=None, avoid_tolls: bool=None, avoid_highways: bool=None, waypoints: str=None, start_time: str=None, avoid_ferries: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find the best route between an origin and a destination, passing through waypoints (if specified)"
    destination: The location to which you wish to calculate directions. 
        origin: The location from which you wish to calculate directions.
        geometry_format: geometry format: latlng, lnglat or geojson
        avoid_tolls: the route should avoid toll roads
        avoid_highways: the route should avoid highways
        waypoints: Specifies an array of intermediate locations to include along the route between the origin and destination points as pass through locations. Max: 23
        start_time: Time when travel is expected to start. You can specify the time as an integer in seconds since midnight, January 1, 1970 UTC or you can use \\\"now\\\" to specify the current time.
        avoid_ferries: the route should avoid ferries
        
    """
    url = f"https://trueway-directions2.p.rapidapi.com/FindDrivingPath"
    querystring = {'destination': destination, 'origin': origin, }
    if geometry_format:
        querystring['geometry_format'] = geometry_format
    if avoid_tolls:
        querystring['avoid_tolls'] = avoid_tolls
    if avoid_highways:
        querystring['avoid_highways'] = avoid_highways
    if waypoints:
        querystring['waypoints'] = waypoints
    if start_time:
        querystring['start_time'] = start_time
    if avoid_ferries:
        querystring['avoid_ferries'] = avoid_ferries
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trueway-directions2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

