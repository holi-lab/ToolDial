import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getall(offset: int, per_page: int, unisex: bool=None, ada: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all restroom records ordered by date descending."
    
    """
    url = f"https://public-bathrooms.p.rapidapi.com/all"
    querystring = {'offset': offset, 'per_page': per_page, }
    if unisex:
        querystring['unisex'] = unisex
    if ada:
        querystring['ada'] = ada
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "public-bathrooms.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbydate(month: str, year: str, day: str, ada: str='false', updated: str='true', unisex: str='false', per_page: str='10', page: str='01', offset: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for restroom records updated or created on or after a given `date`"
    
    """
    url = f"https://public-bathrooms.p.rapidapi.com/date"
    querystring = {'month': month, 'year': year, 'day': day, }
    if ada:
        querystring['ada'] = ada
    if updated:
        querystring['updated'] = updated
    if unisex:
        querystring['unisex'] = unisex
    if per_page:
        querystring['per_page'] = per_page
    if page:
        querystring['page'] = page
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "public-bathrooms.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbyquery(query: str, ada: bool=None, unisex: bool=None, offset: int=0, per_page: int=10, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Perform full-text search of restroom records."
    
    """
    url = f"https://public-bathrooms.p.rapidapi.com/search"
    querystring = {'query': query, }
    if ada:
        querystring['ada'] = ada
    if unisex:
        querystring['unisex'] = unisex
    if offset:
        querystring['offset'] = offset
    if per_page:
        querystring['per_page'] = per_page
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "public-bathrooms.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbylocation(lat: int, lng: int, page: int=1, offset: int=0, ada: bool=None, unisex: bool=None, per_page: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets public bathrooms by Latitude and Longitude"
    
    """
    url = f"https://public-bathrooms.p.rapidapi.com/location"
    querystring = {'lat': lat, 'lng': lng, }
    if page:
        querystring['page'] = page
    if offset:
        querystring['offset'] = offset
    if ada:
        querystring['ada'] = ada
    if unisex:
        querystring['unisex'] = unisex
    if per_page:
        querystring['per_page'] = per_page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "public-bathrooms.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

