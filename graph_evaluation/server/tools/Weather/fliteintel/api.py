import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def latest_tracon_acceptance_rate_tar_for_major_us_tracons(tra: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides the latest TRACON Acceptance Rate (TAR) available for major US Terminal Radar Approach Controls (TRACON). Please Enter the 3 letter TRACON code for your airport."
    
    """
    url = f"https://fliteintel1.p.rapidapi.com/v1/metering/tar/tracon/{tra}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fliteintel1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_runway_acceptance_rate_rar_for_major_us_airports(location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides the latest Runway Acceptance Rate (RAR) available for major US airports. Please Enter the 3 letter domestic location code for your airport."
    
    """
    url = f"https://fliteintel1.p.rapidapi.com/v1/metering/rar/airport/{location}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fliteintel1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_airport_acceptance_rate_aar_for_major_us_airports(location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides the latest Airport Acceptance Rate (AAR) available for major US airports. Please Enter the 3 letter domestic location code for your airport."
    
    """
    url = f"https://fliteintel1.p.rapidapi.com/v1/metering/aar/airport/{location}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fliteintel1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_arrival_airport_configuration_aac_for_major_us_airports(location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides the latest Arrival Airport Configuration (AAC) available for major US airports. Please Enter the 3 letter domestic location code for your airport."
    
    """
    url = f"https://fliteintel1.p.rapidapi.com/v1/metering/aac/airport/{location}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fliteintel1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_taf_for_all_us_airports(location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides latest TAF for the US airport of your choice in iWXXM format - please enter your airport's ICAO code as parameter."
    
    """
    url = f"https://fliteintel1.p.rapidapi.com/v1/weather/taf/us/iwxxm/items/{location}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fliteintel1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_metar_for_all_us_airports(location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides the latest available METAR information for your chosen airport in iWXXM format - pass the ICAO code of the US airport as the parameter to receive the information"
    
    """
    url = f"https://fliteintel1.p.rapidapi.com/v1/weather/metar/us/iwxxm/items/{location}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fliteintel1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_rvr_for_your_airport(location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the latest RVR information available for all runways of your airport - please input your chosen airport's ICAO code as parameter"
    
    """
    url = f"https://fliteintel1.p.rapidapi.com/v1/weather/rvr/us/items/{location}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fliteintel1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_datis_for_your_airport(location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the latest Datis information available for your airport - please input your chosen airport's ICAO code as parameter"
    
    """
    url = f"https://fliteintel1.p.rapidapi.com/v1/weather/datis/us/items/{location}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fliteintel1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_metar_in_tac_format_for_your_airport_all_over_the_world(location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides latest METAR for the airports all over the world in TAC format - please enter your airport's ICAO code as parameter."
    
    """
    url = f"https://fliteintel1.p.rapidapi.com/v1/weather/metar/global/tac/items/{location}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fliteintel1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_taf_in_tac_format_for_your_airport_all_over_the_world(location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides latest TAF for the airports all over the world in TAC format - please enter your airport's ICAO code as parameter."
    
    """
    url = f"https://fliteintel1.p.rapidapi.com/v1/weather/taf/global/tac/items/{location}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fliteintel1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

