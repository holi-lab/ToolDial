import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_by_title(s: str, l: int=None, y: int=None, m: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for Movies or TV Shows by title, include option year or mediatype filter to narrow your results"
    s: Movie or TV Show Title to search
        l: Limit returned items.
You can use numbes from 1 to 100 (default is 50)
        y: Year (format YYYY, for example 2020) to limit title search (would also include previous and next year movies)
        m: Search for only movie or tv show, leave empty for both
        
    """
    url = f"https://mdblist.p.rapidapi.com/"
    querystring = {'s': s, }
    if l:
        querystring['l'] = l
    if y:
        querystring['y'] = y
    if m:
        querystring['m'] = m
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mdblist.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_by_trakt_tv_id(t: str, m: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns movie or TV Show info"
    t: A valid Trakt.tv ID
        m: Mediatype *movie* or *show* (default *movie*)
        
    """
    url = f"https://mdblist.p.rapidapi.com/"
    querystring = {'t': t, }
    if m:
        querystring['m'] = m
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mdblist.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_by_imdb_id(i: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns movie or TV Show info by IMDb ID"
    i: A valid IMDb ID
        
    """
    url = f"https://mdblist.p.rapidapi.com/"
    querystring = {'i': i, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mdblist.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_by_tmdb_id(tm: int, m: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns movie or TV Show info"
    tm: A valid TMDb ID
        m: Mediatype *movie* or *show* (default *movie*)
        
    """
    url = f"https://mdblist.p.rapidapi.com/"
    querystring = {'tm': tm, }
    if m:
        querystring['m'] = m
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mdblist.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_by_tvdb_id(tv: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns TV Show info"
    
    """
    url = f"https://mdblist.p.rapidapi.com/"
    querystring = {'tv': tv, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mdblist.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

