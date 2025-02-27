import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_game_critic_properties(game_name: str, platform: str='pc', reviews: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve information about game and its reviews and scores"
    reviews: Defines if should include recent reviews in properties
        
    """
    url = f"https://metacriticapi.p.rapidapi.com/games/{game_name}"
    querystring = {}
    if platform:
        querystring['platform'] = platform
    if reviews:
        querystring['reviews'] = reviews
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metacriticapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_top_games(platform: str='pc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Receive top scoring games by platform"
    
    """
    url = f"https://metacriticapi.p.rapidapi.com/games/top"
    querystring = {}
    if platform:
        querystring['platform'] = platform
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metacriticapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def health_check(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check server status"
    
    """
    url = f"https://metacriticapi.p.rapidapi.com/status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metacriticapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_music(page: str='1', genre: str='action', filter: str='date', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Receive music list"
    
    """
    url = f"https://metacriticapi.p.rapidapi.com/music"
    querystring = {}
    if page:
        querystring['page'] = page
    if genre:
        querystring['genre'] = genre
    if filter:
        querystring['filter'] = filter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metacriticapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_new_music(filter: str='date', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Receive new released music"
    
    """
    url = f"https://metacriticapi.p.rapidapi.com/music/new"
    querystring = {}
    if filter:
        querystring['filter'] = filter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metacriticapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_new_tv_series(filter: str='date', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Receive new released tv series"
    
    """
    url = f"https://metacriticapi.p.rapidapi.com/tv/new"
    querystring = {}
    if filter:
        querystring['filter'] = filter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metacriticapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_new_movies(filter: str='date', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Receive new released movies"
    
    """
    url = f"https://metacriticapi.p.rapidapi.com/movies/new"
    querystring = {}
    if filter:
        querystring['filter'] = filter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metacriticapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_games(genre: str='action', filter: str='date', platform: str='pc', page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Receive games list"
    
    """
    url = f"https://metacriticapi.p.rapidapi.com/games"
    querystring = {}
    if genre:
        querystring['genre'] = genre
    if filter:
        querystring['filter'] = filter
    if platform:
        querystring['platform'] = platform
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metacriticapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_new_games(platform: str='pc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Receive new realesed games by platform"
    
    """
    url = f"https://metacriticapi.p.rapidapi.com/games/new"
    querystring = {}
    if platform:
        querystring['platform'] = platform
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metacriticapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_tv_series_critic_properties(name: str, reviews: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve information about tv series and its reviews and scores"
    reviews: Defines if should include recent reviews in properties
        
    """
    url = f"https://metacriticapi.p.rapidapi.com/tv/{name}"
    querystring = {}
    if reviews:
        querystring['reviews'] = reviews
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metacriticapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_album_critic_properties(name: str, reviews: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve information about album and its reviews and scores"
    reviews: Defines if should include recent reviews in properties
        
    """
    url = f"https://metacriticapi.p.rapidapi.com/music/{name}"
    querystring = {}
    if reviews:
        querystring['reviews'] = reviews
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metacriticapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_movie_critic_properties(movie_name: str, reviews: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve information about movie and its reviews and scores"
    reviews: Defines if should include recent reviews in properties
        
    """
    url = f"https://metacriticapi.p.rapidapi.com/movies/{movie_name}"
    querystring = {}
    if reviews:
        querystring['reviews'] = reviews
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metacriticapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

