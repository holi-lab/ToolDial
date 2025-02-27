import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def feed_channel(cids: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get episode IDs and release date from one or more channels"
    cids: Channel IDs. Separate by commas
        
    """
    url = f"https://podcast-api1.p.rapidapi.com/episodes/overview"
    querystring = {'cids': cids, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "podcast-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Codes for categories"
    
    """
    url = f"https://podcast-api1.p.rapidapi.com/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "podcast-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_channels(category_id: int, country: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gte the most popular channels by category"
    category_id: /categories endpoint has all category codes
        country: country code
        
    """
    url = f"https://podcast-api1.p.rapidapi.com/top_channels/v2"
    querystring = {'category_id': category_id, }
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "podcast-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def typeahead(keyword: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Term suggestion autocomplete"
    keyword: Terms for searching
        
    """
    url = f"https://podcast-api1.p.rapidapi.com/keywords/suggestion"
    querystring = {'keyword': keyword, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "podcast-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def episode(eid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about an individual episode"
    eid: Episode ID
        
    """
    url = f"https://podcast-api1.p.rapidapi.com/episode/v4"
    querystring = {'eid': eid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "podcast-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channel(cid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about a channel"
    cid: Channel ID
        
    """
    url = f"https://podcast-api1.p.rapidapi.com/channel/v3"
    querystring = {'cid': cid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "podcast-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_channel(keyword: str, limit: int=20, country: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search podcast channels"
    keyword: Terms for searching
        limit: Number of results. Max. 200
        country: country code
        
    """
    url = f"https://podcast-api1.p.rapidapi.com/search_channel/v2"
    querystring = {'keyword': keyword, }
    if limit:
        querystring['limit'] = limit
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "podcast-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def feed_episode(cid: str, eids: str='544642284,542878320', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all episodes of a channel"
    cid: Channel ID
        eids: Select only specific episode IDs. Separate by commas
        
    """
    url = f"https://podcast-api1.p.rapidapi.com/episode_list/v2"
    querystring = {'cid': cid, }
    if eids:
        querystring['eids'] = eids
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "podcast-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_episode(keyword: str, cid: int=None, limit: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search episodes in all channels or in an individual channel"
    keyword: Terms for searching
        cid: Channel ID. Search episodes in this channel.
        limit: < 200
        
    """
    url = f"https://podcast-api1.p.rapidapi.com/search_episode"
    querystring = {'keyword': keyword, }
    if cid:
        querystring['cid'] = cid
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "podcast-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

