import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def airports_database_api(iata_code: str='CDG', icao_code: str='LFPG', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Airport data is fundamental to any aviation-related business.
		Learn how to use the Airport Database API in your products."
    iata_code: Filtering by Airport IATA code.
        icao_code: Filtering by Airport ICAO code.
        
    """
    url = f"https://iatacodes-iatacodes-v1.p.rapidapi.com/api/v9/airports"
    querystring = {}
    if iata_code:
        querystring['iata_code'] = iata_code
    if icao_code:
        querystring['icao_code'] = icao_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iatacodes-iatacodes-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airlines_database_api(icao_code: str='AAL', iata_code: str='AA', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "AirLabs Airline data is the most essential data for any travel startup and companies related to the aviation industry."
    icao_code: Filtering by Airline ICAO code
        iata_code: Filtering by Airline IATA code
        
    """
    url = f"https://iatacodes-iatacodes-v1.p.rapidapi.com/api/v9/airlines"
    querystring = {}
    if icao_code:
        querystring['icao_code'] = icao_code
    if iata_code:
        querystring['iata_code'] = iata_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iatacodes-iatacodes-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cities_database_api(city_code: str='SIN', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "City data complements and enhances airport data, if you need to create features that are independent of airports, you can use cities to replace them, and you can also combine them."
    city_code: Filtering by IATA City code
        
    """
    url = f"https://iatacodes-iatacodes-v1.p.rapidapi.com/api/v9/cities"
    querystring = {}
    if city_code:
        querystring['city_code'] = city_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iatacodes-iatacodes-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def live_flight_schedules_api(dep_icao: str='KMIA', arr_iata: str='SFO', airline_iata: str='BA', arr_icao: str='KSFO', airline_icao: str='BAW', dep_iata: str='MIA', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The live schedules provide an information about the current status of the departure and arrival flights."
    
    """
    url = f"https://iatacodes-iatacodes-v1.p.rapidapi.com/api/v9/schedules"
    querystring = {}
    if dep_icao:
        querystring['dep_icao'] = dep_icao
    if arr_iata:
        querystring['arr_iata'] = arr_iata
    if airline_iata:
        querystring['airline_iata'] = airline_iata
    if arr_icao:
        querystring['arr_icao'] = arr_icao
    if airline_icao:
        querystring['airline_icao'] = airline_icao
    if dep_iata:
        querystring['dep_iata'] = dep_iata
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iatacodes-iatacodes-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def flight_information_api(flight_icao: str='AAL6', flight_iata: str='AA6', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Single flight detailed information by flight number with ICAO or IATA prefix."
    
    """
    url = f"https://iatacodes-iatacodes-v1.p.rapidapi.com/api/v9/flight"
    querystring = {}
    if flight_icao:
        querystring['flight_icao'] = flight_icao
    if flight_iata:
        querystring['flight_iata'] = flight_iata
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iatacodes-iatacodes-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def name_suggestion_api(term: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Travel destination autocomplete API."
    term: **term** or **q** or **query** or **s** or **search** or **text**

Part of the destination name - airport, city or country.
Between 3 and 30 characters
        
    """
    url = f"https://iatacodes-iatacodes-v1.p.rapidapi.com/api/v9/suggest"
    querystring = {'term': term, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iatacodes-iatacodes-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nearby_airports_api(distance: str, lng: str, lat: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use the NearBy API to show the nearest departure airport or all available airports in the desired radius."
    distance: Distance from required Geo location (km)
        lng: Geo Longitude
        lat: Geo Latitude
        
    """
    url = f"https://iatacodes-iatacodes-v1.p.rapidapi.com/api/v9/nearby"
    querystring = {'distance': distance, 'lng': lng, 'lat': lat, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iatacodes-iatacodes-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fleets_database_api(airline_iata: str='AA', airline_icao: str='AAL', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "AirLabs Global aircraft database."
    airline_iata: Filtering by Airline IATA code
        airline_icao: Filtering by Airline ICAO code
        
    """
    url = f"https://iatacodes-iatacodes-v1.p.rapidapi.com/api/v9/fleets"
    querystring = {}
    if airline_iata:
        querystring['airline_iata'] = airline_iata
    if airline_icao:
        querystring['airline_icao'] = airline_icao
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iatacodes-iatacodes-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def routes_database_api(arr_icao: str='OMAA', airline_icao: str='ALK', dep_iata: str='CMB', dep_icao: str='VCBI', arr_iata: str='AUH', airline_iata: str='UL', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Global Airline Routes Database API. 
		Only 1 of the following query parameters is required."
    arr_icao: Arrival Airport ICAO Code
        airline_icao: Airline ICAO Code
        dep_iata: Departure Airport IATA Code
        dep_icao: Departure Airport ICAO Code
        arr_iata: Arrival Airport IATA Code
        airline_iata: Airline IATA Code
        
    """
    url = f"https://iatacodes-iatacodes-v1.p.rapidapi.com/api/v9/routes"
    querystring = {}
    if arr_icao:
        querystring['arr_icao'] = arr_icao
    if airline_icao:
        querystring['airline_icao'] = airline_icao
    if dep_iata:
        querystring['dep_iata'] = dep_iata
    if dep_icao:
        querystring['dep_icao'] = dep_icao
    if arr_iata:
        querystring['arr_iata'] = arr_iata
    if airline_iata:
        querystring['airline_iata'] = airline_iata
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iatacodes-iatacodes-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def real_time_flight_tracking_api(bbox: str='46.01,-12.21,56.84,9.66', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using the Flight API, you can track flights and show them on maps in your own products."
    bbox: Bounding box of your map view
        
    """
    url = f"https://iatacodes-iatacodes-v1.p.rapidapi.com/api/v9/flights"
    querystring = {}
    if bbox:
        querystring['bbox'] = bbox
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iatacodes-iatacodes-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

