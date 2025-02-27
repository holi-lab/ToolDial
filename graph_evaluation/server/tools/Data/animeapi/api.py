import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def all_anime(release: int=None, genre: str=None, pagesize: int=None, page: int=None, q: str='naruto', status: str=None, sortby: str=None, qintitle: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed anime information using search queries, returned in JSON format."
    release: The release year of the anime
        q: Search query to match titles or description of anime
        status: The status of the anime: ongoing or completed
        qintitle: Search query to match titles of anime only
        
    """
    url = f"https://animeapi4.p.rapidapi.com/all"
    querystring = {}
    if release:
        querystring['release'] = release
    if genre:
        querystring['genre'] = genre
    if pagesize:
        querystring['pageSize'] = pagesize
    if page:
        querystring['page'] = page
    if q:
        querystring['q'] = q
    if status:
        querystring['status'] = status
    if sortby:
        querystring['sortBy'] = sortby
    if qintitle:
        querystring['qInTitle'] = qintitle
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animeapi4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

