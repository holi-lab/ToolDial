import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def schneetage(is_from: str, lon: int, lat: int, to: str='2021-09-21', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Beantwortet die Frage, wie viel Schnee und an wie viel Tagen innerhalb eines Zeitraums vorhanden gewesen ist. Wird verwendet zum Beispiel bei der Einsatzplanung im Winterdienst und Hausmeisterservice sowie im Tourismus."
    
    """
    url = f"https://ecoweather.p.rapidapi.com/ucSNOW"
    querystring = {'from': is_from, 'lon': lon, 'lat': lat, }
    if to:
        querystring['to'] = to
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecoweather.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def heizgradtage(is_from: str, lon: int, lat: int, to: str='2021-09-21', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Beantwortet die Frage, wie viele Heizgradtage innerhalb eines Zeitraums vorhanden gewesen sind. Wird verwendet zum Beispiel bei der unterjährigen Abgrenzung von Heizkosten bei der Nebenkostenabrechnung."
    
    """
    url = f"https://ecoweather.p.rapidapi.com/ucHDD"
    querystring = {'from': is_from, 'lon': lon, 'lat': lat, }
    if to:
        querystring['to'] = to
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecoweather.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def niederschlag(is_from: str, lon: int, lat: int, to: str='2021-09-21', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Beantwortet die Frage, wie viel Niederschlag innerhalb eines Zeitraums vorhanden gewesen ist. Wird verwendet zum Beispiel bei der Dimensionierung von Regenwassernutzung (Zisterne) oder bei der Planung von Gärten ."
    
    """
    url = f"https://ecoweather.p.rapidapi.com/ucPREC"
    querystring = {'from': is_from, 'lon': lon, 'lat': lat, }
    if to:
        querystring['to'] = to
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecoweather.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ecoweather(lon: int, lat: int, is_from: str, to: str='2022-09-31', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve historical weather data for a location. Dataset lasts back until year 2016. Results are limited to 366 days (1 year)  per request."
    lon: Longitude of a geo-location in degrees. 
        lat: Latitude of a geo-location in degrees. 
        from: Start date in YYYY-MM-DD format.
        to: End date in YYYY-MM-DD format. 

Note: if time period relative to `from` is more than 366 days it will automatically be replaced with this date.
        
    """
    url = f"https://ecoweather.p.rapidapi.com/ecoweather"
    querystring = {'lon': lon, 'lat': lat, 'from': is_from, }
    if to:
        querystring['to'] = to
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecoweather.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

