import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_countries(keyword: str='Indonesia', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get country list"
    keyword: Use this parameter to filter cities by keyword or set empty if you don't want to use it 
        
    """
    url = f"https://50k-radio-stations.p.rapidapi.com/get/countries"
    querystring = {}
    if keyword:
        querystring['keyword'] = keyword
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "50k-radio-stations.p.rapidapi.com"
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
    country_id: Use this parameter to filter cities by country id or set empty if you don't want to use it 
        keyword: Use this parameter to filter cities by keyword or set empty if you don't want to use it 
        
    """
    url = f"https://50k-radio-stations.p.rapidapi.com/get/cities"
    querystring = {}
    if country_id:
        querystring['country_id'] = country_id
    if keyword:
        querystring['keyword'] = keyword
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "50k-radio-stations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_channels(keyword: str='a', genre_id: int=None, city_id: int=None, country_id: int=50, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get channel list"
    keyword: Use this parameter to filter cities by keyword or set empty if you don't want to use it 
        genre_id: Use this parameter to filter cities by genre id or set empty if you don't want to use it 
        city_id: Use this parameter to filter cities by city id or set empty if you don't want to use it 
        country_id: Use this parameter to filter cities by country id or set empty if you don't want to use it 
        page: Use this parameter to get next page
        
    """
    url = f"https://50k-radio-stations.p.rapidapi.com/get/channels"
    querystring = {}
    if keyword:
        querystring['keyword'] = keyword
    if genre_id:
        querystring['genre_id'] = genre_id
    if city_id:
        querystring['city_id'] = city_id
    if country_id:
        querystring['country_id'] = country_id
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "50k-radio-stations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_genres(keyword: str='music', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get genre list"
    keyword: Use this parameter to filter cities by keyword or set empty if you don't want to use it 
        
    """
    url = f"https://50k-radio-stations.p.rapidapi.com/get/genres"
    querystring = {}
    if keyword:
        querystring['keyword'] = keyword
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "50k-radio-stations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

