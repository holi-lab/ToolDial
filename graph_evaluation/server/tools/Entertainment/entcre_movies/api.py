import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_movies_by_category(q: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Movie By Category 
		All Available Categories
		- Action
		- Adventure
		- Animation
		- Biography
		- Comedy
		- Documentary
		- Drama
		- Fantasy
		- History
		- Horror
		- Sci-Fi
		- Romance"
    
    """
    url = f"https://entcre-movies.p.rapidapi.com/category"
    querystring = {'q': q, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entcre-movies.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_movies_by_ratings(q: int, type: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Movies By Ratings from various movies platforms.
		Plaform Ratings Available
		IMDB **useCase ->** type=im
		RottenTomatoes **useCase ->** type=rt
		Metascore **useCase ->** type=ms
		Film Affinity **useCase ->** type=fm
		TheMovieDb **useCase ->** type=md"
    
    """
    url = f"https://entcre-movies.p.rapidapi.com/ratings"
    querystring = {'q': q, 'type': type, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entcre-movies.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_movies_by_year(q: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search  list of movies by year"
    
    """
    url = f"https://entcre-movies.p.rapidapi.com/year"
    querystring = {'q': q, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entcre-movies.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_movie_by_id(mid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Movie by Entcre mId"
    
    """
    url = f"https://entcre-movies.p.rapidapi.com/movies/id"
    querystring = {'mId': mid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entcre-movies.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_movies_by_cast(q: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Movies By Cast"
    
    """
    url = f"https://entcre-movies.p.rapidapi.com/movies/cast"
    querystring = {'q': q, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entcre-movies.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_top_actor_and_actress(q: str, ty: str='name', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Top Actors and Actress by name and castId. Search Default is by name"
    
    """
    url = f"https://entcre-movies.p.rapidapi.com/cast"
    querystring = {'q': q, }
    if ty:
        querystring['ty'] = ty
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entcre-movies.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_casts(p: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List Top Actors and Actress"
    
    """
    url = f"https://entcre-movies.p.rapidapi.com/casts"
    querystring = {}
    if p:
        querystring['p'] = p
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entcre-movies.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_movies_by_title(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter any movie name to search"
    q: Enter any movie name to search
        
    """
    url = f"https://entcre-movies.p.rapidapi.com/title"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "entcre-movies.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

