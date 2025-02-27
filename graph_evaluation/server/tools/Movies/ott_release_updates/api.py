import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def posts_previousreleasess(x_rapidapi_key: str='6bd08cfc6emsha317c32f9167266p194e65jsn982e896efb11', x_rapidapi_host: str='ott-release-updates.p.rapidapi.com', sort: str='vote_average', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Sort the results by IMDB rating/date released
		Query parameters:
		Sorting: sort = date_released/vote_average
		
		These parameters can be applied to all requests"
    
    """
    url = f"https://ott-release-updates.p.rapidapi.com/posts/previousreleasess"
    querystring = {}
    if x_rapidapi_key:
        querystring['X-RapidAPI-Key'] = x_rapidapi_key
    if x_rapidapi_host:
        querystring['X-RapidAPI-Host'] = x_rapidapi_host
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ott-release-updates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def posts_633c599d546dda989d90b2ff(x_rapidapi_host: str='ott-release-updates.p.rapidapi.com', x_rapidapi_key: str='6bd08cfc6emsha317c32f9167266p194e65jsn982e896efb11', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get speicifc movie information by Movie Id"
    
    """
    url = f"https://ott-release-updates.p.rapidapi.com/posts/633c599d546dda989d90b2ff"
    querystring = {}
    if x_rapidapi_host:
        querystring['X-RapidAPI-Host'] = x_rapidapi_host
    if x_rapidapi_key:
        querystring['X-RapidAPI-Key'] = x_rapidapi_key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ott-release-updates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def posts_nextweeks(x_rapidapi_host: str='ott-release-updates.p.rapidapi.com', x_rapidapi_key: str='6bd08cfc6emsha317c32f9167266p194e65jsn982e896efb11', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all Movies Which are going to stream in the coming weeks
		in all major OTT platform"
    
    """
    url = f"https://ott-release-updates.p.rapidapi.com/posts/nextweeks"
    querystring = {}
    if x_rapidapi_host:
        querystring['X-RapidAPI-Host'] = x_rapidapi_host
    if x_rapidapi_key:
        querystring['X-RapidAPI-Key'] = x_rapidapi_key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ott-release-updates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def posts_search(x_rapidapi_key: str='6bd08cfc6emsha317c32f9167266p194e65jsn982e896efb11', x_rapidapi_host: str='ott-release-updates.p.rapidapi.com', moviename: str='bi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search from the movie list by movie name / some characters in the movie name
		Query parameters:
		Search: movieName = 'bim'"
    
    """
    url = f"https://ott-release-updates.p.rapidapi.com/posts/search"
    querystring = {}
    if x_rapidapi_key:
        querystring['X-RapidAPI-Key'] = x_rapidapi_key
    if x_rapidapi_host:
        querystring['X-RapidAPI-Host'] = x_rapidapi_host
    if moviename:
        querystring['movieName'] = moviename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ott-release-updates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def posts(x_rapidapi_host: str='ott-release-updates.p.rapidapi.com', x_rapidapi_key: str='6bd08cfc6emsha317c32f9167266p194e65jsn982e896efb11', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all Movies"
    
    """
    url = f"https://ott-release-updates.p.rapidapi.com/posts"
    querystring = {}
    if x_rapidapi_host:
        querystring['X-RapidAPI-Host'] = x_rapidapi_host
    if x_rapidapi_key:
        querystring['X-RapidAPI-Key'] = x_rapidapi_key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ott-release-updates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def posts_previousreleases(x_rapidapi_key: str='6bd08cfc6emsha317c32f9167266p194e65jsn982e896efb11', x_rapidapi_host: str='ott-release-updates.p.rapidapi.com', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all Movies Which are streaming currently in all major OTT platform"
    
    """
    url = f"https://ott-release-updates.p.rapidapi.com/posts/previousreleases"
    querystring = {}
    if x_rapidapi_key:
        querystring['X-RapidAPI-Key'] = x_rapidapi_key
    if x_rapidapi_host:
        querystring['X-RapidAPI-Host'] = x_rapidapi_host
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ott-release-updates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

