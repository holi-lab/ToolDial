import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_postal_code(postalcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for listings matching the specified postal code. This can be either numbers only, e.g. "1013", or a full postal code, e.g. "1013 BC". Case sensitive."
    
    """
    url = f"https://roofhunt1.p.rapidapi.com/rentals/search/postalcode"
    querystring = {'postalCode': postalcode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "roofhunt1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_address(address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for listings that contain the specified address. Case sensitive."
    
    """
    url = f"https://roofhunt1.p.rapidapi.com/rentals/search/address"
    querystring = {'address': address, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "roofhunt1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(city: str, pricemaximum: int=None, roomsminimum: int=None, areamaximum: int=None, priceminimum: int=None, areaminimum: int=None, roomsmaximum: int=None, page: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search our database of tens of thousands of listings throughout The Netherlands."
    
    """
    url = f"https://roofhunt1.p.rapidapi.com/rentals/search"
    querystring = {'city': city, }
    if pricemaximum:
        querystring['priceMaximum'] = pricemaximum
    if roomsminimum:
        querystring['roomsMinimum'] = roomsminimum
    if areamaximum:
        querystring['areaMaximum'] = areamaximum
    if priceminimum:
        querystring['priceMinimum'] = priceminimum
    if areaminimum:
        querystring['areaMinimum'] = areaminimum
    if roomsmaximum:
        querystring['roomsMaximum'] = roomsmaximum
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "roofhunt1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def recent_listings(city: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Quickly get the 20 most recent listings in the specified city."
    
    """
    url = f"https://roofhunt1.p.rapidapi.com/rentals/{city}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "roofhunt1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

