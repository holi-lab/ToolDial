import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def places_in_radius(location_types: str, distance: str, starting_point: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns list of locations that meet following criteria:
		- are within defined radius from starting point (radius is in meters), which is a set of coordinates specified in
		format: starting_point=54.373639725729085,18.61484334290751 (lat, lng)
		- belong to defined categories (eg. 'grocery_store', 'supermarket', 'pharmacy')
		
		Reponse contains following data:
		- place_id (Google place ID - unique identifier assigned to each place on Google Maps)
		- name (location name)
		- location data (tuple of coordinates)
		- type of the place
		- distance data tuple, which contains walking and driving time to location from starting point"
    location_types: List of location types.

Example locations:
grocery_store
supermarket
store
restaurant
railway_station
bus_station
gym
        distance: Distance from origin (starting_point) in **meters**

Example: 1000 means 1km or ~0.621371 mile

        starting_point: Starting point coordinates (latitude,longitude)
These are example valid coordinates:

37.81995483709157,-122.47833251953125 (Golden Gate Bridge)
36.16644125323845,-115.14111136959748 (Somewhere in Downtown Las Vegas)


        
    """
    url = f"https://places-in-radius.p.rapidapi.com/places_in_radius"
    querystring = {'location_types': location_types, 'distance': distance, 'starting_point': starting_point, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "places-in-radius.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def openapi_openapi_json_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://places-in-radius.p.rapidapi.com/openapi.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "places-in-radius.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

