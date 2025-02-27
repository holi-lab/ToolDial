import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def trending_on_appletv(geo: str, types: str='movie', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint will return a blended list of trending content from across VOD streaming services."
    geo: Supported GEOs: GB, US (default) 
        types: Multi-value parameter, pipe separated. 
Supported values: series, movie 
Default: series|movie
        
    """
    url = f"https://vod-trends.p.rapidapi.com/apple-tv-plus-trending"
    querystring = {'geo': geo, }
    if types:
        querystring['types'] = types
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vod-trends.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trending_on_itunes(geo: str, types: str='movie', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint will return a list of trending content from iTunes."
    geo: Supported GEOs: GB, US (default) 
        types: Multi-value parameter, pipe separated. 
Supported values: series, movie 
Default: series|movie
        
    """
    url = f"https://vod-trends.p.rapidapi.com/itunes-trending"
    querystring = {'geo': geo, }
    if types:
        querystring['types'] = types
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vod-trends.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trending_on_amazon_prime(geo: str, types: str='movie', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint will return a blended list of trending content from across VOD streaming services."
    geo: Supported GEOs: GB, US (default) 
        types: Multi-value parameter, pipe separated. 
Supported values: series, movie 
Default: series|movie
        
    """
    url = f"https://vod-trends.p.rapidapi.com/amazon-prime-trending"
    querystring = {'geo': geo, }
    if types:
        querystring['types'] = types
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vod-trends.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trending_on_amazon_video(geo: str, types: str='movie', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint will return a list of trending content from Amazon Video."
    geo: Supported GEOs: GB, US (default) 
        types: Multi-value parameter, pipe separated. 
Supported values: series, movie 
Default: series|movie
        
    """
    url = f"https://vod-trends.p.rapidapi.com/amazon-video-trending"
    querystring = {'geo': geo, }
    if types:
        querystring['types'] = types
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vod-trends.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trending_on_hbo_max(geo: str, types: str='movie', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint will return a blended list of trending content from across VOD streaming services."
    geo: Supported GEOs: GB, US (default) 
        types: Multi-value parameter, pipe separated. 
Supported values: series, movie 
Default: series|movie
        
    """
    url = f"https://vod-trends.p.rapidapi.com/hbo-max-trending"
    querystring = {'geo': geo, }
    if types:
        querystring['types'] = types
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vod-trends.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trending_on_hulu(geo: str, types: str='movie', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint will return a blended list of trending content from across VOD streaming services."
    geo: Supported GEOs: GB, US (default) 
        types: Multi-value parameter, pipe separated. 
Supported values: series, movie 
Default: series|movie
        
    """
    url = f"https://vod-trends.p.rapidapi.com/hulu-trending"
    querystring = {'geo': geo, }
    if types:
        querystring['types'] = types
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vod-trends.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trending_on_netflix(geo: str, types: str='movie', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint will return a blended list of trending content from across VOD streaming services."
    geo: Supported GEOs: GB, US (default) 
        types: Multi-value parameter, pipe separated. 
Supported values: series, movie 
Default: series|movie
        
    """
    url = f"https://vod-trends.p.rapidapi.com/netflix-trending"
    querystring = {'geo': geo, }
    if types:
        querystring['types'] = types
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vod-trends.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trending_on_disney(geo: str, types: str='movie', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint will return a blended list of trending content from across VOD streaming services."
    geo: Supported GEOs: GB, US (default) 
        types: Multi-value parameter, pipe separated. 
Supported values: series, movie 
Default: series|movie
        
    """
    url = f"https://vod-trends.p.rapidapi.com/disney-plus-trending"
    querystring = {'geo': geo, }
    if types:
        querystring['types'] = types
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vod-trends.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trending_on_peacock(geo: str, types: str='movie', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint will return a blended list of trending content from across VOD streaming services."
    geo: Supported GEOs: GB, US (default) 
        types: Multi-value parameter, pipe separated. 
Supported values: series, movie 
Default: series|movie
        
    """
    url = f"https://vod-trends.p.rapidapi.com/peacock-trending"
    querystring = {'geo': geo, }
    if types:
        querystring['types'] = types
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vod-trends.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trending_on_showtime_anytime(geo: str, types: str='movie', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint will return a blended list of trending content from across VOD streaming services."
    geo: Supported GEOs: GB, US (default) 
        types: Multi-value parameter, pipe separated. 
Supported values: series, movie 
Default: series|movie
        
    """
    url = f"https://vod-trends.p.rapidapi.com/showtime-anytime-trending"
    querystring = {'geo': geo, }
    if types:
        querystring['types'] = types
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vod-trends.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trending_on_csb_all_access(geo: str, types: str='movie', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint will return a blended list of trending content from across VOD streaming services."
    geo: Supported GEOs: GB, US (default) 
        types: Multi-value parameter, pipe separated. 
Supported values: series, movie 
Default: series|movie
        
    """
    url = f"https://vod-trends.p.rapidapi.com/cbs-all-access-trending"
    querystring = {'geo': geo, }
    if types:
        querystring['types'] = types
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vod-trends.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trending_content(geo: str, types: str='movie', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint will return a blended list of trending content from across VOD streaming services."
    geo: Supported GEOs: GB, US (default) 
        types: Multi-value parameter, pipe separated. 
Supported values: series, movie 
Default: series|movie
        
    """
    url = f"https://vod-trends.p.rapidapi.com/trending"
    querystring = {'geo': geo, }
    if types:
        querystring['types'] = types
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vod-trends.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

