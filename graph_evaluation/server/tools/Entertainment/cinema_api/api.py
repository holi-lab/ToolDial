import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_movie_reviews_by_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Endpoint returns a complete list of reviews associated to the ID. You can obtain the ID from the Get Movie By ID endpoint."
    id: Movie ID. You can obtain the ID from the Get Movie By ID endpoint.
        
    """
    url = f"https://cinema-api.p.rapidapi.com/get_reviews/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cinema-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_movie_images_by_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Endpoint returns a complete list of images associated to the ID. You can obtain the ID from the Get Movie By ID endpoint."
    id: Movie ID. You can obtain the ID from the Get Movie By ID endpoint.
        
    """
    url = f"https://cinema-api.p.rapidapi.com/get_images/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cinema-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_movie_id_by_title(category: str, title: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to retrieve the ID of the movies you are interested in. You can search your movie or your preferite actor."
    category: Two categories allowed:
*movies*
*actors*
        title: Enter the text to search. 
        
    """
    url = f"https://cinema-api.p.rapidapi.com/get_ids/{title}/{category}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cinema-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_cast_by_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With this endpoint, you can retrieve the full cast's list (together with their images and their role in the movie)."
    id: Movie ID. You can obtain the ID from the Get Movie By ID endpoint.
        
    """
    url = f"https://cinema-api.p.rapidapi.com/get_cast/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cinema-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

