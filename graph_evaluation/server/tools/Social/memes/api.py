import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def list_of_memes(genre: str='memes', type: str='top', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get List of Memes from genre
		
		**Default Type:**  top
		**Available Type**
		- Top
		- Hot
		- New
		
		**Default Genre: ** memes
		**Available Genres**
		- "anime"
		- "dnd"
		- "history"
		- "historyani"
		- "memes"
		- "politics"
		- "science"
		- "lotr"
		- "ww2""
    
    """
    url = f"https://memes9.p.rapidapi.com/api/list"
    querystring = {}
    if genre:
        querystring['genre'] = genre
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "memes9.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_meme(genre: str='memes', type: str='top', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Random Meme from genre
		
		**Default Type:**  top
		**Available Type**
		- Top
		- Hot
		- New
		
		**Default Genre: ** memes
		**Available Genres**
		- "anime"
		- "dnd"
		- "history"
		- "historyani"
		- "memes"
		- "politics"
		- "science"
		- "lotr"
		- "ww2""
    
    """
    url = f"https://memes9.p.rapidapi.com/api/random"
    querystring = {}
    if genre:
        querystring['genre'] = genre
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "memes9.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

