import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_movies_by_cast_name(cast_name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<br /><br />
		<b>Authentication:</b> not required"
    
    """
    url = f"https://imdb_api4.p.rapidapi.com/get_movies_by_cast_name"
    querystring = {}
    if cast_name:
        querystring['cast_name'] = cast_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imdb_api4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_movies_by_director(movie_director: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<br /><br />
		<b>Authentication:</b> not required"
    
    """
    url = f"https://imdb_api4.p.rapidapi.com/get_movies_by_director"
    querystring = {}
    if movie_director:
        querystring['movie_director'] = movie_director
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imdb_api4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_movies_by_name(movie_name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<br /><br />
		<b>Authentication:</b> not required"
    
    """
    url = f"https://imdb_api4.p.rapidapi.com/get_movies_by_name"
    querystring = {}
    if movie_name:
        querystring['Movie_name'] = movie_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imdb_api4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_movies_by_year(movie_year: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<br /><br />
		<b>Authentication:</b> not required"
    
    """
    url = f"https://imdb_api4.p.rapidapi.com/get_movies_by_year"
    querystring = {}
    if movie_year:
        querystring['movie_year'] = movie_year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imdb_api4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

