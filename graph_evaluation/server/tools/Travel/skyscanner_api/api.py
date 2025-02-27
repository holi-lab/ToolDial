import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def locations(locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Geographical locations have the concept of hierarchy. Each location has a parent of a larger hierarchy.  I.e.: a place `type` of `airport` can have a parent of `city` and `city` can have a parent of `country`."
    locale: Locale of the request. 
List of locales can be retrieved from the `Locales` endpoint.

        
    """
    url = f"https://skyscanner-api.p.rapidapi.com/v3/geo/hierarchy/flights/{locale}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def locales(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can use the `/locales` endpoint to retrieve the locales that we support to translate your content. The names of the locales returned are in the native language associated with the locale."
    
    """
    url = f"https://skyscanner-api.p.rapidapi.com/v3/culture/locales"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def markets(locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can use the `/markets` endpoint to retrieve the market countries that we support. Most suppliers (airlines, travel agents, and car hire dealers) set their fares based on the market (or country of purchase). It is therefore necessary to specify the market country in every query. The names of the markets returned are localised based on a locale passed as a parameter."
    locale: Locale of the request.
List of locales can be retrieved from the `Locales` endpoint.

        
    """
    url = f"https://skyscanner-api.p.rapidapi.com/v3/culture/markets/{locale}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def currencies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can use the `/currencies` endpoint to retrieve the currencies that we support and information about how we format them in Skyscanner."
    
    """
    url = f"https://skyscanner-api.p.rapidapi.com/v3/culture/currencies"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def carriers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Carriers API returns a full list of active carriers with name and IATA code indexed by their carrierId."
    
    """
    url = f"https://skyscanner-api.p.rapidapi.com/v3/flights/carriers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

