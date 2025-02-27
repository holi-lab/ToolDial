import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def directions_between_2_locations(start_lon: int, end_lon: int, start_lat: int, end_lat: int, distance_unit: str='km', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns information about the route between two locations in terms of distance, duration, and steps.
		
		Below Example: **Directions from Sfax to Tunis**"
    start_lon: Longitude of the starting point (required)
        end_lon: Longitude of the ending point (required)
        start_lat: Latitude of the starting point (required)
        end_lat: Latitude of the ending point (required)
        distance_unit: Unit of distance measurement, valid options are **km** (default), and **mi**.
        
    """
    url = f"https://tunisia-api.p.rapidapi.com/directions"
    querystring = {'start_lon': start_lon, 'end_lon': end_lon, 'start_lat': start_lat, 'end_lat': end_lat, }
    if distance_unit:
        querystring['distance_unit'] = distance_unit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tunisia-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def measure_distance(lon2: int, lat2: int, lon1: int, lat1: int, unit: str='km', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint calculates the distance between two locations based on their latitude and longitude coordinates, while allowing the user to specify the unit of measurement.
		
		Below Example: **Distance from Sfax to Tunis**"
    lon2: Longitude of the second location (required)
        lat2: Latitude of the second location (required)
        lon1: Longitude of the first location (required)
        lat1: Latitude of the first location (required)
        unit: Unit of distance measurement, valid options are **km** (default), **mi**, **ft**, and **yd**.

        
    """
    url = f"https://tunisia-api.p.rapidapi.com/distance"
    querystring = {'lon2': lon2, 'lat2': lat2, 'lon1': lon1, 'lat1': lat1, }
    if unit:
        querystring['unit'] = unit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tunisia-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reverse_geocode(lon: str, lat: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to perform reverse geocoding in Tunisia by providing query parameters for latitude and longitude. It returns the name of the city where the location is located."
    lon: The longitude of the location.
        lat: The latitude of the location.
        
    """
    url = f"https://tunisia-api.p.rapidapi.com/georev"
    querystring = {'lon': lon, 'lat': lat, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tunisia-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def facilities_lookup(city: str, province: str, type: str, limit: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to get facilities in Tunisia like hospital, bank, college, etc. by providing optional query parameters for facility type, province and city. It returns a list of facilities that match the query parameters."
    city: The city where the facility is located
        province: The region where the facility is located
        type: The type of amenity facility to search for (default: **hospital**)
Options:
**aerodrome, bar, cafe, fast_food, pub, restaurant, college, driving_school, school, university, bank, atm, pharmacy,** etc..        

[More options->](https://wiki.openstreetmap.org/wiki/Map_features#Amenity)
        limit: The number of facilities to query.
        
    """
    url = f"https://tunisia-api.p.rapidapi.com/facility"
    querystring = {'city': city, 'province': province, 'type': type, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tunisia-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def geocode(address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to lookup locations in Tunisia by providing an address query parameter. It returns the latitude, longitude and city name of the location."
    address: The name of the location to look up.
        
    """
    url = f"https://tunisia-api.p.rapidapi.com/geofy"
    querystring = {'address': address, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tunisia-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

