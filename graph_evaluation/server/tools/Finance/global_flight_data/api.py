import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def united_states_scheduled_flights_growth(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get weekly United States scheduled departing flights growth."
    
    """
    url = f"https://global-flight-data.p.rapidapi.com/api/v1/resources/scheduled-flights-growth-us-weekly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def united_states_scheduled_flights_level(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get weekly United States scheduled departing flights level."
    
    """
    url = f"https://global-flight-data.p.rapidapi.com/api/v1/resources/scheduled-flights-level-us-weekly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def australia_scheduled_flights_growth(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get weekly Australia scheduled departing flights growth."
    
    """
    url = f"https://global-flight-data.p.rapidapi.com/api/v1/resources/scheduled-flights-growth-australia-weekly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def australia_scheduled_flights_level(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get weekly Australia scheduled departing flights level."
    
    """
    url = f"https://global-flight-data.p.rapidapi.com/api/v1/resources/scheduled-flights-level-australia-weekly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def china_scheduled_flights_growth(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get weekly China scheduled departing flights growth."
    
    """
    url = f"https://global-flight-data.p.rapidapi.com/api/v1/resources/scheduled-flights-growth-china-weekly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def china_scheduled_flights_level(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get weekly China scheduled departing flights level."
    
    """
    url = f"https://global-flight-data.p.rapidapi.com/api/v1/resources/scheduled-flights-level-china-weekly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def france_scheduled_flights_level(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get weekly France scheduled departing flights level."
    
    """
    url = f"https://global-flight-data.p.rapidapi.com/api/v1/resources/scheduled-flights-level-france-weekly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def germany_scheduled_flights_growth(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get weekly Germany scheduled departing flights growth."
    
    """
    url = f"https://global-flight-data.p.rapidapi.com/api/v1/resources/scheduled-flights-growth-germany-weekly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def germany_scheduled_flights_level(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get weekly Germany scheduled departing flights level."
    
    """
    url = f"https://global-flight-data.p.rapidapi.com/api/v1/resources/scheduled-flights-level-germany-weekly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hong_kong_scheduled_flights_growth(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get weekly Hong Kong scheduled departing flights growth."
    
    """
    url = f"https://global-flight-data.p.rapidapi.com/api/v1/resources/scheduled-flights-growth-hong-kong-weekly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hong_kong_scheduled_flights_level(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get weekly Hong Kong scheduled departing flights level."
    
    """
    url = f"https://global-flight-data.p.rapidapi.com/api/v1/resources/scheduled-flights-level-hong-kong-weekly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def india_scheduled_flights_growth(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get weekly India scheduled departing flights growth."
    
    """
    url = f"https://global-flight-data.p.rapidapi.com/api/v1/resources/scheduled-flights-growth-india-weekly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def india_scheduled_flights_level(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get weekly India scheduled departing flights level."
    
    """
    url = f"https://global-flight-data.p.rapidapi.com/api/v1/resources/scheduled-flights-level-india-weekly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def france_scheduled_flights_growth(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get weekly France scheduled departing flights growth."
    
    """
    url = f"https://global-flight-data.p.rapidapi.com/api/v1/resources/scheduled-flights-growth-france-weekly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def italy_scheduled_flights_growth(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get weekly Italy scheduled departing flights growth."
    
    """
    url = f"https://global-flight-data.p.rapidapi.com/api/v1/resources/scheduled-flights-growth-italy-weekly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def italy_scheduled_flights_level(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get weekly Italy scheduled departing flights level."
    
    """
    url = f"https://global-flight-data.p.rapidapi.com/api/v1/resources/scheduled-flights-level-italy-weekly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def japan_scheduled_flights_growth(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get weekly Japan scheduled departing flights growth."
    
    """
    url = f"https://global-flight-data.p.rapidapi.com/api/v1/resources/scheduled-flights-growth-japan-weekly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def japan_scheduled_flights_level(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get weekly Japan scheduled departing flights level."
    
    """
    url = f"https://global-flight-data.p.rapidapi.com/api/v1/resources/scheduled-flights-level-japan-weekly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def singapore_scheduled_flights_growth(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get weekly Singapore scheduled departing flights growth."
    
    """
    url = f"https://global-flight-data.p.rapidapi.com/api/v1/resources/scheduled-flights-growth-singapore-weekly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def singapore_scheduled_flights_level(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get weekly Singapore scheduled departing flights level."
    
    """
    url = f"https://global-flight-data.p.rapidapi.com/api/v1/resources/scheduled-flights-level-singapore-weekly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def south_korea_scheduled_flights_growth(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get weekly South Korea scheduled departing flights growth."
    
    """
    url = f"https://global-flight-data.p.rapidapi.com/api/v1/resources/scheduled-flights-growth-south-korea-weekly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def south_korea_scheduled_flights_level(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get weekly South Korea scheduled departing flights level."
    
    """
    url = f"https://global-flight-data.p.rapidapi.com/api/v1/resources/scheduled-flights-level-south-korea-weekly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def spain_scheduled_flights_growth(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get weekly Spain scheduled departing flights growth."
    
    """
    url = f"https://global-flight-data.p.rapidapi.com/api/v1/resources/scheduled-flights-growth-spain-weekly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def spain_scheduled_flights_level(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get weekly Spain scheduled departing flights level."
    
    """
    url = f"https://global-flight-data.p.rapidapi.com/api/v1/resources/scheduled-flights-level-spain-weekly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sweden_scheduled_flights_growth(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get weekly Sweden scheduled departing flights growth."
    
    """
    url = f"https://global-flight-data.p.rapidapi.com/api/v1/resources/scheduled-flights-growth-sweden-weekly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sweden_scheduled_flights_level(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get weekly Sweden scheduled departing flights level."
    
    """
    url = f"https://global-flight-data.p.rapidapi.com/api/v1/resources/scheduled-flights-level-sweden-weekly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def uae_scheduled_flights_growth(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get weekly United Arab Emirates scheduled departing flights growth."
    
    """
    url = f"https://global-flight-data.p.rapidapi.com/api/v1/resources/scheduled-flights-growth-uae-weekly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def uae_scheduled_flights_level(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get weekly United Arab Emirates scheduled departing flights level."
    
    """
    url = f"https://global-flight-data.p.rapidapi.com/api/v1/resources/scheduled-flights-level-uae-weekly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def united_kingdom_scheduled_flights_growth(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get weekly United Kingdom scheduled departing flights growth."
    
    """
    url = f"https://global-flight-data.p.rapidapi.com/api/v1/resources/scheduled-flights-growth-uk-weekly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def united_kingdom_scheduled_flights_level(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get weekly United Kingdom scheduled departing flights level."
    
    """
    url = f"https://global-flight-data.p.rapidapi.com/api/v1/resources/scheduled-flights-level-uk-weekly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

