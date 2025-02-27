import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_all_trending_data(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "this endpoint returns relevant tranding data relevant to following keyword
		keyword
		other
		anime
		music
		applications
		television
		games
		movies"
    
    """
    url = f"https://torrent-scrapper-api.p.rapidapi.com/top/all"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "torrent-scrapper-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_by_title(key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "this endpoint return relevant data for a given key"
    
    """
    url = f"https://torrent-scrapper-api.p.rapidapi.com/search/{key}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "torrent-scrapper-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_trending_movie_anime_music_applications_television_movies_games_others(topkey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "this endpoint return trending data for following category:
		other
		anime
		music
		applications
		television
		games
		movies"
    
    """
    url = f"https://torrent-scrapper-api.p.rapidapi.com/top/{topkey}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "torrent-scrapper-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_today_popular_movies_anime_tv_games_music_apps_other_copy(popkey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "this endpoint return popular data about particular category for today like - movies/anime/tv/games/music/apps/other"
    
    """
    url = f"https://torrent-scrapper-api.p.rapidapi.com/popular/{popkey}/day"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "torrent-scrapper-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_week_popular_movies_anime_tv_games_music_apps_other(popkey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "this endpoint return popular data about particular category for current week like - movies/anime/tv/games/music/apps/other"
    
    """
    url = f"https://torrent-scrapper-api.p.rapidapi.com/popular/{popkey}/week"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "torrent-scrapper-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

