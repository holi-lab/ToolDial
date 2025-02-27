import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_distance(xlongitude: int, xlatitude: int, stype: str='LARGE_AIRPORT,MEDIUM_AIRPORT', ndistance: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search airports arround a geo point.
		Specify Latitude, Longitude. You can specify a distance, by default it's 30km arround."
    stype: You can shoose what airports you want to liste :
one or more values separated by coma :
SMALL_AIRPORT
MEDIUM_AIRPORT
LARGE_AIRPORT
CLOSED
SEAPLANE_BASE
BALLOONPORT
HELIPORT
        
    """
    url = f"https://iata-icao-airport-heliport.p.rapidapi.com/SearchProxiAirport/{xlatitude}/{xlongitude}/{ndistance}/{stype}"
    querystring = {}
    if stype:
        querystring['sType'] = stype
    if ndistance:
        querystring['nDistance'] = ndistance
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iata-icao-airport-heliport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_by_icao(icao: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search airport by ICAO code"
    
    """
    url = f"https://iata-icao-airport-heliport.p.rapidapi.com/SearchCodeICAO/{icao}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iata-icao-airport-heliport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_by_iata_code(iata: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search airport by IATA code"
    iata: Airport IATA Code (3 letters) 
        
    """
    url = f"https://iata-icao-airport-heliport.p.rapidapi.com/SearchCodeIATA/{iata}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iata-icao-airport-heliport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

