import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def genres(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of Netfilx movie genres"
    
    """
    url = f"https://netflix-unofficial.p.rapidapi.com/api/genres"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "netflix-unofficial.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def directors(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of directors in Netflix movies"
    
    """
    url = f"https://netflix-unofficial.p.rapidapi.com/api/directors"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "netflix-unofficial.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def actors(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of actors in Netflix movies"
    
    """
    url = f"https://netflix-unofficial.p.rapidapi.com/api/actors"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "netflix-unofficial.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(year: str=None, director: str=None, time_min: str=None, rating: str=None, genre: str=None, time_max: str=None, actors: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for movies on Netflix with multiple filters. Take the capital letter into account.
		The parameters we use are optional and are as follows:
		- year: year of the film
		- actors: actor or actor of the film
		- genre: genre of the film
		- director: film director
		- rating: minimum IMDB rating
		- time_min: minimum movie time
		- time_max: maximum time of the film."
    
    """
    url = f"https://netflix-unofficial.p.rapidapi.com/api/search"
    querystring = {}
    if year:
        querystring['year'] = year
    if director:
        querystring['director'] = director
    if time_min:
        querystring['time_min'] = time_min
    if rating:
        querystring['rating'] = rating
    if genre:
        querystring['genre'] = genre
    if time_max:
        querystring['time_max'] = time_max
    if actors:
        querystring['actors'] = actors
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "netflix-unofficial.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countries(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of Netflix production countries"
    
    """
    url = f"https://netflix-unofficial.p.rapidapi.com/api/countries"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "netflix-unofficial.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

