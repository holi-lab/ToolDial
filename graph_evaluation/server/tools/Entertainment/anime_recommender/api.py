import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_auto_complete(search: str, amnt: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Auto Complete, as name implies, auto complete of a search, returns up to the desired amount."
    
    """
    url = f"https://anime-recommender.p.rapidapi.com/auto_complete"
    querystring = {'search': search, 'amnt': amnt, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "anime-recommender.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_anime_title_exists(anime_title: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Boolean to represent if anime title exists"
    
    """
    url = f"https://anime-recommender.p.rapidapi.com/title_exists"
    querystring = {'anime_title': anime_title, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "anime-recommender.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_anime_info(anime_title: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get info on a specific anime
		
		returns general ready-to-use info on an anime."
    
    """
    url = f"https://anime-recommender.p.rapidapi.com/get_anime_info"
    querystring = {'anime_title': anime_title, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "anime-recommender.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_anime_titles(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Anime titles"
    
    """
    url = f"https://anime-recommender.p.rapidapi.com/get_titles"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "anime-recommender.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_recommendation(anime_title: str, number_of_anime: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Basic Recommendation Request
		Takes : Name of Anime + amount"
    
    """
    url = f"https://anime-recommender.p.rapidapi.com/"
    querystring = {'anime_title': anime_title, 'number_of_anime': number_of_anime, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "anime-recommender.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

