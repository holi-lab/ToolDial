import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_flights_by_airline(airline: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return list of live flights by the given airline code.
		Visit our [demo site](https://core-api.net/flight/map.html)."
    airline: International airline code in ICAO format.
e.g SIA for Singapore Airline
        
    """
    url = f"https://flight-data4.p.rapidapi.com/get_airline_flights"
    querystring = {'airline': airline, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flight-data4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_aircraft_photo(image: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Pass the image name which return by "Get Flight Info" method as parameter to get aircraft photo.
		*Image might not be display correctly here due to base64 issue of Test Endpoint, but it work indeed.*"
    image: Image name return by "Get Flight Info" method.
        
    """
    url = f"https://flight-data4.p.rapidapi.com/get_aircraft_photo"
    querystring = {'image': image, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flight-data4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_flights_in_area(longitude: int, latitude: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return all flights that within 500 km in radius by the given coordinates.
		Visit our [demo site](https://core-api.net/flight/map.html)."
    
    """
    url = f"https://flight-data4.p.rapidapi.com/get_area_flights"
    querystring = {'longitude': longitude, 'latitude': latitude, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flight-data4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_flight_info(flight: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return live flight information for given IATA number.
		Visit our [demo site](https://core-api.net/flight/map.html)."
    flight: Flight number in IATA format.
        
    """
    url = f"https://flight-data4.p.rapidapi.com/get_flight_info"
    querystring = {'flight': flight, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flight-data4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_flights(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return all current globally active flights on earth.
		Visit our [demo site](https://core-api.net/flight/map.html)."
    
    """
    url = f"https://flight-data4.p.rapidapi.com/get_all_flights"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flight-data4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

