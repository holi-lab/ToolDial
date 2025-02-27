import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_list_of_movies_sorted_by_imdb_rating_with_parameters(limit: str='8', select: str='name', page: str='5', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the list of movies sorted by their IMDb Rating. Query parameters like limiting, pagination and selection can be added."
    
    """
    url = f"https://moviehut-random-movie.p.rapidapi.com/api/movies"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if select:
        querystring['select'] = select
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moviehut-random-movie.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_movie_details_by_name(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns name, releaseYear, certificate, certificate, genre, imdbRating, overview, metaScore, and director parameters for a movie of your choice."
    
    """
    url = f"https://moviehut-random-movie.p.rapidapi.com/api/movie/{name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moviehut-random-movie.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_random_movie(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint will return a random movie with all the details such as name, releaseYear, certificate, certificate, genre, imdbRating, overview, metaScore, and director."
    
    """
    url = f"https://moviehut-random-movie.p.rapidapi.com/api/random"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moviehut-random-movie.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

