import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_genres(keyword: str='jap', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get genre list"
    
    """
    url = f"https://60k-radio-stations.p.rapidapi.com/get/genres"
    querystring = {}
    if keyword:
        querystring['keyword'] = keyword
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "60k-radio-stations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_channels(page: int=None, genre_id: int=None, city_id: int=None, country_id: int=None, keyword: str='He', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get channel list"
    
    """
    url = f"https://60k-radio-stations.p.rapidapi.com/get/channels"
    querystring = {}
    if page:
        querystring['page'] = page
    if genre_id:
        querystring['genre_id'] = genre_id
    if city_id:
        querystring['city_id'] = city_id
    if country_id:
        querystring['country_id'] = country_id
    if keyword:
        querystring['keyword'] = keyword
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "60k-radio-stations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_countries(keyword: str='Aus', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get country list"
    
    """
    url = f"https://60k-radio-stations.p.rapidapi.com/get/countries"
    querystring = {}
    if keyword:
        querystring['keyword'] = keyword
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "60k-radio-stations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_cities(country_id: int=None, keyword: str='Jakarta', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get city list"
    
    """
    url = f"https://60k-radio-stations.p.rapidapi.com/get/cities"
    querystring = {}
    if country_id:
        querystring['country_id'] = country_id
    if keyword:
        querystring['keyword'] = keyword
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "60k-radio-stations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

