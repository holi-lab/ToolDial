import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_all(page: str, size: str, types: str=None, genres: str='Fantasy,Drama', sortby: str='ranking', sortorder: str='asc', search: str='Fullmetal', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get List of anime. You can search, sort an specify genre"
    types: Anime type, separated by comma
        genres: genres separated by comma
        sortby: ranking or title
        sortorder: asc or desc
        search: Search by title or alternative titles. Search will ignore sort
        
    """
    url = f"https://anime-db.p.rapidapi.com/anime"
    querystring = {'page': page, 'size': size, }
    if types:
        querystring['types'] = types
    if genres:
        querystring['genres'] = genres
    if sortby:
        querystring['sortBy'] = sortby
    if sortorder:
        querystring['sortOrder'] = sortorder
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "anime-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_by_id(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get anime by its corresponding ID.
		
		ID is got from 'Get All' endpoint"
    
    """
    url = f"https://anime-db.p.rapidapi.com/anime/by-id/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "anime-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_genres(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Genres"
    
    """
    url = f"https://anime-db.p.rapidapi.com/genre"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "anime-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_one_anime_by_ranking(rank: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get anime by ranking"
    
    """
    url = f"https://anime-db.p.rapidapi.com/anime/by-ranking/{rank}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "anime-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

