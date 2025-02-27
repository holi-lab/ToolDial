import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_torrents(keywords: str, quantity: int, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get downloadable  torrent link by movie name."
    quantity: MAX:40
        
    """
    url = f"https://movie-tv-music-search-and-download.p.rapidapi.com/search"
    querystring = {'keywords': keywords, 'quantity': quantity, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "movie-tv-music-search-and-download.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_monthly_top_100_music_torrents(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Monthly Top 100 Music Torrents"
    
    """
    url = f"https://movie-tv-music-search-and-download.p.rapidapi.com/monthly_top100_music"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "movie-tv-music-search-and-download.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_monthly_top_100_games_torrents(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Monthly Top 100 Games Torrents"
    
    """
    url = f"https://movie-tv-music-search-and-download.p.rapidapi.com/monthly_top100_games"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "movie-tv-music-search-and-download.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_monthly_top_100_tv_shows_torrents(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Monthly Top 100 TV-shows Torrents"
    
    """
    url = f"https://movie-tv-music-search-and-download.p.rapidapi.com/monthly_top100_tv_shows"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "movie-tv-music-search-and-download.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_monthly_top_100_movies_torrents(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Monthly Top 100 Movies Torrents"
    
    """
    url = f"https://movie-tv-music-search-and-download.p.rapidapi.com/monthly_top100_movies"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "movie-tv-music-search-and-download.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

