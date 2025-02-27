import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getsilverpricehistory(no_of_days: str='365', place: str='lucknow', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the silver price history for a specific location up to 1800 days. A "place" parameter must be included in the request to specify the desired location, and a "no_of_days" parameter must be included to specify the number of days of history to retrieve. The response is returned in JSON format and includes the silver price history for the specified location and number of days."
    
    """
    url = f"https://gold-silver-live-prices.p.rapidapi.com/getSilverPriceHistory"
    querystring = {}
    if no_of_days:
        querystring['no_of_days'] = no_of_days
    if place:
        querystring['place'] = place
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gold-silver-live-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsilverrate(place: str='chennai', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the current silver rate for a specific location. A "place" parameter must be included in the request to specify the desired location. The response is returned in JSON format and includes the current silver rate for the specified location."
    
    """
    url = f"https://gold-silver-live-prices.p.rapidapi.com/getSilverRate"
    querystring = {}
    if place:
        querystring['place'] = place
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gold-silver-live-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getgoldpricehistory(no_of_days: int=100, place: str='hyderabad', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the gold price history for a specific location up to 1800 days. A "place" parameter must be included in the request to specify the desired location, and a "no_of_days" parameter must be included to specify the number of days of history to retrieve. The response is returned in JSON format and includes the gold price history for the specified location and number of days."
    
    """
    url = f"https://gold-silver-live-prices.p.rapidapi.com/getGoldPriceHistory"
    querystring = {}
    if no_of_days:
        querystring['no_of_days'] = no_of_days
    if place:
        querystring['place'] = place
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gold-silver-live-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getgoldrate(place: str='kalyan', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the current gold rate for a specific location. A "place" parameter must be included in the request to specify the desired location. The response is returned in JSON format and includes the current gold rate for the specified location."
    
    """
    url = f"https://gold-silver-live-prices.p.rapidapi.com/getGoldRate"
    querystring = {}
    if place:
        querystring['place'] = place
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gold-silver-live-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getallplaces(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a list of all the available locations for which gold and silver prices can be retrieved. The response is returned in JSON format and includes a list of all the locations."
    
    """
    url = f"https://gold-silver-live-prices.p.rapidapi.com/getAllPlaces"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gold-silver-live-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

