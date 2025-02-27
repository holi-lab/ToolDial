import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def stations(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of stations"
    
    """
    url = f"https://flixbus.p.rapidapi.com/v1/stations"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flixbus.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_trips(to_id: int, from_id: int, currency: str, departure_date: str, number_adult: int, number_bike_slot: int=0, search_by: str='cities', number_children: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search trips from-to. Specify for which parameters (from_id, to_id) you are looking for, city or station"
    to_id: To id
        from_id: From id
        departure_date: Departure date
        number_adult: Number of adults
        number_bike_slot: Number of bike slot
        search_by: Search by
        number_children: Number of children
        
    """
    url = f"https://flixbus.p.rapidapi.com/v1/search-trips"
    querystring = {'to_id': to_id, 'from_id': from_id, 'currency': currency, 'departure_date': departure_date, 'number_adult': number_adult, }
    if number_bike_slot:
        querystring['number_bike_slot'] = number_bike_slot
    if search_by:
        querystring['search_by'] = search_by
    if number_children:
        querystring['number_children'] = number_children
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flixbus.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cities(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of cities"
    
    """
    url = f"https://flixbus.p.rapidapi.com/v1/cities"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flixbus.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def schedule(station_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of schedule"
    station_id: Station id
        
    """
    url = f"https://flixbus.p.rapidapi.com/v1/schedule"
    querystring = {'station_id': station_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flixbus.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trip_details(trip_uid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a trip details by trip uid"
    trip_uid: Trip uid
        
    """
    url = f"https://flixbus.p.rapidapi.com/v1/trip-details"
    querystring = {'trip_uid': trip_uid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flixbus.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

