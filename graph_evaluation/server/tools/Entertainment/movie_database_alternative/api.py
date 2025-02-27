import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def by_search(s: str, y: str=None, page: str='1', type: str=None, r: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search movie titles"
    
    """
    url = f"https://movie-database-alternative.p.rapidapi.com/"
    querystring = {'s': s, }
    if y:
        querystring['y'] = y
    if page:
        querystring['page'] = page
    if type:
        querystring['type'] = type
    if r:
        querystring['r'] = r
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "movie-database-alternative.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def by_id_or_title(type: str=None, i: str='tt4154796', plot: str=None, callback: str=None, y: str=None, r: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for Movies by ID or Title"
    type: Type of result to return: (movie, series, episode)
        i: A valid IMDb ID (e.g. tt4154796)
        plot: Return short or full plot: (short, full)
        callback: JSONP callback name
        y: Year of release (e.g. 2019)
        r: The data type to return: (json, xml)
        
    """
    url = f"https://movie-database-alternative.p.rapidapi.com/"
    querystring = {}
    if type:
        querystring['type'] = type
    if i:
        querystring['i'] = i
    if plot:
        querystring['plot'] = plot
    if callback:
        querystring['callback'] = callback
    if y:
        querystring['y'] = y
    if r:
        querystring['r'] = r
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "movie-database-alternative.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

