import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def greatest_of_all_time_songs_of_the_summer(range: str='1-10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the Greatest of All Time Songs of the Summer chart."
    range: Returns all results if range is not specified
        
    """
    url = f"https://billboard3.p.rapidapi.com/greatest-of-all-time-songs-of-the-summer"
    querystring = {}
    if range:
        querystring['range'] = range
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def greatest_of_all_time_artists(range: str='1-10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the Greatest of All Time Artists chart."
    range: Returns all results if range is not specified
        
    """
    url = f"https://billboard3.p.rapidapi.com/greatest-of-all-time-artists"
    querystring = {}
    if range:
        querystring['range'] = range
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def greatest_of_all_time_hot_100_songs(range: str='1-10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the Greatest of All Time Hot 100 Songs chart."
    range: Returns all results if range is not specified
        
    """
    url = f"https://billboard3.p.rapidapi.com/greatest-hot-100-singles"
    querystring = {}
    if range:
        querystring['range'] = range
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def year_end_top_artists(year: int, range: str='1-10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the Year-End Top Artists chart."
    range: Returns all results if range is not specified
        
    """
    url = f"https://billboard3.p.rapidapi.com/year-end-top-artists"
    querystring = {'year': year, }
    if range:
        querystring['range'] = range
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def year_end_billboard_global_200(year: int, range: str='1-10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the Year-End Billboard Global 200 chart."
    range: Returns all results if range is not specified
        
    """
    url = f"https://billboard3.p.rapidapi.com/year-end-global-200"
    querystring = {'year': year, }
    if range:
        querystring['range'] = range
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def year_end_billboard_200_albums(year: int, range: str='1-10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the Year-End Billboard 200 Albums chart."
    range: Returns all results if range is not specified
        
    """
    url = f"https://billboard3.p.rapidapi.com/year-end-top-billboard-200-albums"
    querystring = {'year': year, }
    if range:
        querystring['range'] = range
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def year_end_hot_100_songs(year: int, range: str='1-10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the Year-End Hot 100 Songs chart."
    range: Returns all results if range is not specified
        
    """
    url = f"https://billboard3.p.rapidapi.com/hot-100-songs"
    querystring = {'year': year, }
    if range:
        querystring['range'] = range
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def billboard_global_200(date: str, range: str='1-10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the Billboard Global 200 chart."
    range: Returns all results if range is not specified
        
    """
    url = f"https://billboard3.p.rapidapi.com/billboard-global-200"
    querystring = {'date': date, }
    if range:
        querystring['range'] = range
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def artist_100(date: str, range: str='1-10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the Artist 100 chart."
    range: Returns all results if range is not specified
        
    """
    url = f"https://billboard3.p.rapidapi.com/artist-100"
    querystring = {'date': date, }
    if range:
        querystring['range'] = range
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def billboard_200(date: str, range: str='1-10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the Billboard 200 chart."
    range: Returns all results if range is not specified
        
    """
    url = f"https://billboard3.p.rapidapi.com/billboard-200"
    querystring = {'date': date, }
    if range:
        querystring['range'] = range
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def billboard_hot_100(date: str, range: str='1-10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the Billboard Hot 100 chart."
    range: Returns all results if range is not specified
        
    """
    url = f"https://billboard3.p.rapidapi.com/hot-100"
    querystring = {'date': date, }
    if range:
        querystring['range'] = range
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

