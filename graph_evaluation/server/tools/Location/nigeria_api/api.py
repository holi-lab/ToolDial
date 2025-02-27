import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def directions_between_2_locations(end_lat: int, start_lat: int, start_lon: int, end_lon: int, distance_unit: str='km', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns information about the route between two locations in terms of distance, duration, and steps.
		
		Below Example: **Directions from Kaduna to Maiduguri**"
    end_lat: Latitude of the ending point (required)
        start_lat: Latitude of the starting point (required)
        start_lon: Longitude of the starting point (required)
        end_lon: Longitude of the ending point (required)
        distance_unit: Unit of distance measurement, valid options are **km** (default), and **mi**.
        
    """
    url = f"https://nigeria-api.p.rapidapi.com/directions"
    querystring = {'end_lat': end_lat, 'start_lat': start_lat, 'start_lon': start_lon, 'end_lon': end_lon, }
    if distance_unit:
        querystring['distance_unit'] = distance_unit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nigeria-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def measure_distance(lat1: int, lon1: int, lon2: int, lat2: int, unit: str='km', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint calculates the distance between two locations based on their latitude and longitude coordinates, while allowing the user to specify the unit of measurement.
		
		Below Example: **Distance between Lagos and Abuja**"
    lat1: Latitude of the first location (required)
        lon1: Longitude of the first location (required)
        lon2: Longitude of the second location (required)
        lat2: Latitude of the second location (required)
        unit: Unit of distance measurement, valid options are **km** (default), **mi**, **ft**, and **yd**.

        
    """
    url = f"https://nigeria-api.p.rapidapi.com/distance"
    querystring = {'lat1': lat1, 'lon1': lon1, 'lon2': lon2, 'lat2': lat2, }
    if unit:
        querystring['unit'] = unit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nigeria-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reverse_geocode(lat: str, lon: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to perform reverse geocoding in Nigeria by providing query parameters for latitude and longitude. It returns the name of the city where the location is located."
    lat: The latitude of the location.
        lon: The longitude of the location.
        
    """
    url = f"https://nigeria-api.p.rapidapi.com/georev"
    querystring = {'lat': lat, 'lon': lon, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nigeria-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def facilities_lookup(state: str, type: str, limit: str='10', city: str='Lagos', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to get facilities in Nigeria like hospital, bank, college, etc. by providing optional query parameters for facility type, state and city. It returns a list of facilities that match the query parameters."
    state: The state where the facility is located
        type: The type of amenity facility to search for (default: **hospital**)
Options:
**aerodrome, bar, cafe, fast_food, pub, restaurant, college, driving_school, school, university, bank, atm, pharmacy,** etc..        

[More options->](https://wiki.openstreetmap.org/wiki/Map_features#Amenity)
        limit: The number of facilities to query.
        city: The city where the facility is located
        
    """
    url = f"https://nigeria-api.p.rapidapi.com/facility"
    querystring = {'state': state, 'type': type, }
    if limit:
        querystring['limit'] = limit
    if city:
        querystring['city'] = city
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nigeria-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def geocode(address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to lookup locations in Nigeria by providing an address query parameter. It returns the latitude, longitude and city name of the location."
    address: Name of address.
        
    """
    url = f"https://nigeria-api.p.rapidapi.com/geofy"
    querystring = {'address': address, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nigeria-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

