import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getaircraft(modes: str='A98909', registration: str='N713UW', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint if you are looking for details or statistics for a specific aircraft (find by Mode-S hex code or tail number)"
    modes: Mode-S code
        registration: Registration
        
    """
    url = f"https://radarbox-on-demand.p.rapidapi.com/aircraft"
    querystring = {}
    if modes:
        querystring['modeS'] = modes
    if registration:
        querystring['registration'] = registration
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "radarbox-on-demand.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettaf(icaocode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "TAF (Terminal Aerodrome Forecast) is a weather forecast information service provided by airports to serve the surrounding air traffic. TAF reports are updated several times throughout the day to ensure that pilots have access to the most up-to-date information as possible. You can use this endpoint to get the latest weather information for a specific airport."
    icaocode: ICAO code
        
    """
    url = f"https://radarbox-on-demand.p.rapidapi.com/airports/{icaocode}/taf"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "radarbox-on-demand.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbusinessflightsstatistics(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get statistics for all business flights"
    
    """
    url = f"https://radarbox-on-demand.p.rapidapi.com/statistics/flights/business"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "radarbox-on-demand.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcommercialairportstatistics(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get statistics for commercial flights on major airports"
    
    """
    url = f"https://radarbox-on-demand.p.rapidapi.com/statistics/airports/commercial"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "radarbox-on-demand.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getflightsstatistics(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get statistics for all flights worldwide"
    
    """
    url = f"https://radarbox-on-demand.p.rapidapi.com/statistics/flights"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "radarbox-on-demand.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbusinessairportstatistics(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get statistics for business flights on major airports"
    
    """
    url = f"https://radarbox-on-demand.p.rapidapi.com/statistics/airports/business"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "radarbox-on-demand.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getnotams(icaocode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "NOTAMs (NOtice To AirMen) are a way of local aviation authorities alerting pilots of potential hazards. Use this endpoint if you want to see the alerts that are relevant for a specific airport."
    icaocode: ICAO code
        
    """
    url = f"https://radarbox-on-demand.p.rapidapi.com/airspace/{icaocode}/notams"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "radarbox-on-demand.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmetar(icaocode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "METARs (METeorological Aerodrome Reports) provide a report of the current weather conditions in the vicinity of an aerodrome. Use this endpoint to get the most up-to-date METAR information."
    icaocode: ICAO code
        
    """
    url = f"https://radarbox-on-demand.p.rapidapi.com/airports/{icaocode}/metar"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "radarbox-on-demand.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcommercialflightsstatistics(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get statistics for all commercial flights"
    
    """
    url = f"https://radarbox-on-demand.p.rapidapi.com/statistics/flights/commercial"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "radarbox-on-demand.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbusjetoperatorstatisticsinternal(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get statistics for business jet operators"
    
    """
    url = f"https://radarbox-on-demand.p.rapidapi.com/statistics/flights/business/operators"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "radarbox-on-demand.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmajorairportstatistics(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get statistics for a specific airport"
    
    """
    url = f"https://radarbox-on-demand.p.rapidapi.com/statistics/airports"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "radarbox-on-demand.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getairport(icaocode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint if you know exactly which airport you need to know details for - you can search by ICAO or IATA code."
    icaocode: ICAO code
        
    """
    url = f"https://radarbox-on-demand.p.rapidapi.com/airports/{icaocode}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "radarbox-on-demand.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getairlinesstatistics(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get statistics for major airlines"
    
    """
    url = f"https://radarbox-on-demand.p.rapidapi.com/statistics/airlines"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "radarbox-on-demand.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdatis(icaocode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "ATIS (Automatic Terminal Information Service) is a service that continuously broadcasts aeronautical information around the airport. D-ATIS is the text transcription of the information provided by it. Use this endpoint to find the latest information for a specific airport."
    icaocode: ICAO code
        
    """
    url = f"https://radarbox-on-demand.p.rapidapi.com/airports/{icaocode}/datis"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "radarbox-on-demand.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbusjetmodelsstatisticsinternal(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get statistics for business jet models"
    
    """
    url = f"https://radarbox-on-demand.p.rapidapi.com/statistics/flights/business/models"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "radarbox-on-demand.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getnattracks(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The North Atlantic Tracks are high altitude routes across the atlantic between Western Europe and North America. The tracks are updated daily. Use this endpoint to get the most up-to-date tracks"
    
    """
    url = f"https://radarbox-on-demand.p.rapidapi.com/airspace/natTracks"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "radarbox-on-demand.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def searchaircraft(aircrafttype: str='B738', airline: str='AAL', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint if you want to list all aircraft of a specific ICAO type or company"
    aircrafttype: ICAO aircraft type
        airline: Airline (ICAO code)
        
    """
    url = f"https://radarbox-on-demand.p.rapidapi.com/aircraft/search"
    querystring = {}
    if aircrafttype:
        querystring['aircraftType'] = aircrafttype
    if airline:
        querystring['airline'] = airline
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "radarbox-on-demand.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def searchairport(latitude2: int=None, longitude1: int=None, city: str='New York', onlymajor: bool=None, countryname: str='United States', icaocode: str='KJFK', latitude1: int=None, countrycode: str='US', longitude2: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint if you are trying to find all airports in a country, city or within a specific region."
    latitude2: Latitude 2
        longitude1: Longitude 1
        city: City
        onlymajor: Only major airports
        countryname: Country name
        icaocode: Airport ICAO code
        latitude1: Latitude 1
        countrycode: Country code (2 letters)
        longitude2: Longitude 2
        
    """
    url = f"https://radarbox-on-demand.p.rapidapi.com/airports/search"
    querystring = {}
    if latitude2:
        querystring['latitude2'] = latitude2
    if longitude1:
        querystring['longitude1'] = longitude1
    if city:
        querystring['city'] = city
    if onlymajor:
        querystring['onlyMajor'] = onlymajor
    if countryname:
        querystring['countryName'] = countryname
    if icaocode:
        querystring['icaoCode'] = icaocode
    if latitude1:
        querystring['latitude1'] = latitude1
    if countrycode:
        querystring['countryCode'] = countrycode
    if longitude2:
        querystring['longitude2'] = longitude2
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "radarbox-on-demand.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpactracks(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Pacific Organized Track System describes the set of routes across the Pacific between Hawaii or North America and Japan/South-east Asia - updated on a daily basis. Use this endpoint to get the most up-to-date information."
    
    """
    url = f"https://radarbox-on-demand.p.rapidapi.com/airspace/pacTracks"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "radarbox-on-demand.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

