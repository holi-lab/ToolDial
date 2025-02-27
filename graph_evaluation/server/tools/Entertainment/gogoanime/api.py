import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def top_airing(page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "returns a list of top airing anime"
    page: page limit [1-26]. -1 to fetch all the pages avaliable **Note: Waiting time will be much longer.**
        
    """
    url = f"https://gogoanime2.p.rapidapi.com/top-airing"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gogoanime2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def anime_movies(page: int=1, aph: str='A', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of anime movies"
    page: page limit may vary
        aph: by default the movie list is random. values are from [A-Z]. And 0 is Ascending order with page limit of [1-89].
        
    """
    url = f"https://gogoanime2.p.rapidapi.com/anime-movies"
    querystring = {}
    if page:
        querystring['page'] = page
    if aph:
        querystring['aph'] = aph
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gogoanime2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def episode_thread(episodeid: str, page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "returns a list of comments for that episode"
    page: comments page
        
    """
    url = f"https://gogoanime2.p.rapidapi.com/thread/{episodeid}"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gogoanime2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def streaming_urls(episodeid: str, server: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "returns m3u8 streaming sources."
    server: Currently there're 2 supported servers: `vidcdn`, and `streamsb`.
        
    """
    url = f"https://gogoanime2.p.rapidapi.com/{server}/watch/{episodeid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gogoanime2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def anime_details(animeid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "returns the anime details (including the episodes)"
    
    """
    url = f"https://gogoanime2.p.rapidapi.com/anime-details/{animeid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gogoanime2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def anime_genres(genre: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "returns a list of anime"
    page: The page limit varies by genre
        
    """
    url = f"https://gogoanime2.p.rapidapi.com/genre/{genre}"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gogoanime2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def popular_anime(page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of popular anime"
    page: page limit: [1-504]
        
    """
    url = f"https://gogoanime2.p.rapidapi.com/popular"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gogoanime2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def recent_episodes(type: str='1', page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of recent episodes"
    type: by default the type is `1`. type `1`: japanese with subtitle. type `2`: english dub with no subtitles. type `3`: chinese with english subtitles.
        page: type 1 page limit: [1-331]. type 2: [1-139]. type 3: [1-22].
        
    """
    url = f"https://gogoanime2.p.rapidapi.com/recent-release"
    querystring = {}
    if type:
        querystring['type'] = type
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gogoanime2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

