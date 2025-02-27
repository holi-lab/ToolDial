import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def flights_list_in_boundary(bl_lng: int, tr_lat: int, bl_lat: int, tr_lng: int, type: str=None, speed: str=None, airline: str=None, altitude: str=None, airport: str=None, reg: str=None, limit: int=300, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Helps to list flights, aircrafts in a GEO bounding box, and display them on a map"
    bl_lng: The bottom left longitude of the bounding box
        tr_lat: The top right latitude of the bounding box
        bl_lat: The bottom left latitude of the bounding box
        tr_lng: The top right longitude of the bounding box
        type: The value of 'Code' field returned in .../aircrafts/list endpoint
        speed: Filter by speed. The format is min,max . Ex : 0,460
        airline: The value of 'ICAO' field returned in .../airlines/list endpoint
        altitude: Filter by altitude. The format is min,max . Ex : 0,48000
        airport: The value of 'icao' field returned in .../airports/list endpoint
        reg: Registration Ex : D-AIHV
        limit: The number of flights per response (max 300)
        
    """
    url = f"https://flight-radar1.p.rapidapi.com/flights/list-in-boundary"
    querystring = {'bl_lng': bl_lng, 'tr_lat': tr_lat, 'bl_lat': bl_lat, 'tr_lng': tr_lng, }
    if type:
        querystring['type'] = type
    if speed:
        querystring['speed'] = speed
    if airline:
        querystring['airline'] = airline
    if altitude:
        querystring['altitude'] = altitude
    if airport:
        querystring['airport'] = airport
    if reg:
        querystring['reg'] = reg
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flight-radar1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aircrafts_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List available aircrafts"
    
    """
    url = f"https://flight-radar1.p.rapidapi.com/aircrafts/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flight-radar1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def flights_list_by_airline(airline: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List flights by airline"
    airline: The value of 'icao' field returned in .../airlines/list or .../flights/detail endpoint
The value of 'operator' field returned in .../flights/search endpoint

        
    """
    url = f"https://flight-radar1.p.rapidapi.com/flights/list-by-airline"
    querystring = {'airline': airline, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flight-radar1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def flights_get_playback(flightid: str, timestamp: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get history traveling data of specific flight"
    flightid: Value of 'id' OR 'identification/id' field returned in .../flights/detail or .../flights/get-more-info endpoint
        timestamp: Value of 'departure' field returned in .../flights/detail OR .../flights/get-more-info endpoint
        
    """
    url = f"https://flight-radar1.p.rapidapi.com/flights/get-playback"
    querystring = {'flightId': flightid, 'timestamp': timestamp, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flight-radar1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def flights_get_more_info(query: str, fetchby: str, page: int=1, limit: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get more information of specific aircraft, flight, airline, etc..."
    query: The value of registration (if fetchBy is 'reg') or default (if fetchBy is 'flight') field returned in .../flights/detail
        fetchby: One of the following : reg|flight
* reg is used to get specific aircraft info, flight is used to get specific flight or airline info
        
    """
    url = f"https://flight-radar1.p.rapidapi.com/flights/get-more-info"
    querystring = {'query': query, 'fetchBy': fetchby, }
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flight-radar1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def flights_detail(flight: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed information of specific flight"
    flight: The value of id, flight_id field returned in .../flights/list-in-boundary or .../flights/list-most-tracked endpoint
        
    """
    url = f"https://flight-radar1.p.rapidapi.com/flights/detail"
    querystring = {'flight': flight, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flight-radar1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def flights_list_most_tracked(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the most tracked flights around the world"
    
    """
    url = f"https://flight-radar1.p.rapidapi.com/flights/list-most-tracked"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flight-radar1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def flights_search(query: str, limit: int=25, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for specific flight"
    query: Aircraft number, flight code, airline code, etc...
        limit: The number of items per response (max 25)
        
    """
    url = f"https://flight-radar1.p.rapidapi.com/flights/search"
    querystring = {'query': query, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flight-radar1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airlines_get_logos(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get logos of airlines"
    
    """
    url = f"https://flight-radar1.p.rapidapi.com/airlines/get-logos"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flight-radar1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airlines_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all airlines around the world"
    
    """
    url = f"https://flight-radar1.p.rapidapi.com/airlines/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flight-radar1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airports_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all airports around the world"
    
    """
    url = f"https://flight-radar1.p.rapidapi.com/airports/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flight-radar1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

