import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def reverse_geocoding(lon: int, lat: int, apikey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Translates a coordinate as input into its postal address."
    
    """
    url = f"https://mymappi.p.rapidapi.com/geocoding/reverse"
    querystring = {'lon': lon, 'lat': lat, 'apikey': apikey, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mymappi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def direct_geocoding(q: str, apikey: str, source_lon: int=-3, source_lat: int=40, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Translates an address as input into geographic coordinates. If there are multiple possible results, it retrieves the list of possible results ordered by the distance to the provided source coordinate (if any), in ascending order."
    
    """
    url = f"https://mymappi.p.rapidapi.com/geocoding/direct"
    querystring = {'q': q, 'apikey': apikey, }
    if source_lon:
        querystring['source_lon'] = source_lon
    if source_lat:
        querystring['source_lat'] = source_lat
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mymappi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def snap_to_road(lat: int, lon: int, apikey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given a geographical coordinate, it retrieves the another coordinate which is snapped to the nearest road. This means that the snapped coordinate is found by calculating the intersection between the longitudinal axis of the nearest road segment and the perpendicular line between the provided coordinate and this longitudinal axis."
    
    """
    url = f"https://mymappi.p.rapidapi.com/roads/snap"
    querystring = {'lat': lat, 'lon': lon, 'apikey': apikey, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mymappi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def highway_type(lat: int, lon: int, apikey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given a coordinate, finds the nearest road and determines what type of road it is (motorway, path, primary... etc.)"
    
    """
    url = f"https://mymappi.p.rapidapi.com/roads/highway-type"
    querystring = {'lat': lat, 'lon': lon, 'apikey': apikey, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mymappi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def speed_limit(lat: int, apikey: str, lon: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given a coordinate, it retrieves speed limit info about the nearest road segment, in km/h."
    
    """
    url = f"https://mymappi.p.rapidapi.com/roads/speed-limit"
    querystring = {'lat': lat, 'apikey': apikey, 'lon': lon, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mymappi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def route_calculation(dest: str, profile: str, apikey: str, orig: str, wps: str='40.416906,-3.678286;40.420252,-3.673561;40.426746,-3.671467', steps: bool=None, alternatives: bool=None, geometries: str='polyline', overview: str='simplified', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculates a route between an origin and a destination, possibly passing through many waypoints. It takes into consideration several configuration options in order to customize the response."
    
    """
    url = f"https://mymappi.p.rapidapi.com/directions/route/{profile}"
    querystring = {'dest': dest, 'apikey': apikey, 'orig': orig, }
    if wps:
        querystring['wps'] = wps
    if steps:
        querystring['steps'] = steps
    if alternatives:
        querystring['alternatives'] = alternatives
    if geometries:
        querystring['geometries'] = geometries
    if overview:
        querystring['overview'] = overview
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mymappi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def traveling_salesman(coordinates: str, apikey: str, profile: str, overview: str='simplified', roundtrip: bool=None, destination: str='any', source: str='any', geometries: str='polyline', steps: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "It solves the Traveling Salesman Problem using a greedy heuristic (farthest-insertion algorithm) for 10 or more waypoints and uses brute force for less than 10 waypoints. The returned path does not have to be the fastest path. As TSP is NP-hard it only returns an approximation. Note that all input coordinates have to be connected for this service to work."
    
    """
    url = f"https://mymappi.p.rapidapi.com/directions/traveling-salesman/{profile}/{coordinates}"
    querystring = {'apikey': apikey, }
    if overview:
        querystring['overview'] = overview
    if roundtrip:
        querystring['roundtrip'] = roundtrip
    if destination:
        querystring['destination'] = destination
    if source:
        querystring['source'] = source
    if geometries:
        querystring['geometries'] = geometries
    if steps:
        querystring['steps'] = steps
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mymappi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def isochrone(lon: int, max_time: str, profile: str, lat: int, apikey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Isochrone API allows you to request polygon or line features that show areas that are reachable within a few specified amounts of time from a certain location in different routing profiles (car, on foot and soon bike and public transport)."
    
    """
    url = f"https://mymappi.p.rapidapi.com/directions/isochrone/{profile}"
    querystring = {'lon': lon, 'max_time': max_time, 'lat': lat, 'apikey': apikey, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mymappi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def distance_matrix(profile: str, coordinates: str, apikey: str, destinations: str='3;4;5', annotations: str='duration,distance', sources: str='0;1;2', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This services takes as input a list of coordinates and computes in a matrix format the fastest travel time among all possible pair combinations among all the provided coordinates. If sources/destinations params are provided, those indicate a semi-colon separated list of indices that specify which of the provided coordinates should be included as sources or destinations."
    
    """
    url = f"https://mymappi.p.rapidapi.com/directions/matrix/{profile}/{coordinates}"
    querystring = {'apikey': apikey, }
    if destinations:
        querystring['destinations'] = destinations
    if annotations:
        querystring['annotations'] = annotations
    if sources:
        querystring['sources'] = sources
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mymappi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def transit(dest: str, arrive_by: bool, orig: str, apikey: str, max_walk_distance: str='500', time: str='1:02pm', date: str='04-21-2020', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculates a route between an origin and a destination in public transport. It takes into consideration several configuration options in order to customize the response."
    
    """
    url = f"https://mymappi.p.rapidapi.com/directions/transit"
    querystring = {'dest': dest, 'arrive_by': arrive_by, 'orig': orig, 'apikey': apikey, }
    if max_walk_distance:
        querystring['max_walk_distance'] = max_walk_distance
    if time:
        querystring['time'] = time
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mymappi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_places(lat: int, radius: int, apikey: str, lon: int, next: str='VFdwVlBRPT0=', limit: int=25, type: str='bar', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Searches around a set of coordinates within a certain radius (in meters) to retrieve a list of nearby points of interest of a specific type (optionally)."
    
    """
    url = f"https://mymappi.p.rapidapi.com/places/search"
    querystring = {'lat': lat, 'radius': radius, 'apikey': apikey, 'lon': lon, }
    if next:
        querystring['next'] = next
    if limit:
        querystring['limit'] = limit
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mymappi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

