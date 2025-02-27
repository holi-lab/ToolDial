import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def movie_info(is_id: str, mode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Showing complete information about the movie"
    
    """
    url = f"https://aiom-db-all-in-one-movie-database.p.rapidapi.com/detail/"
    querystring = {'id': is_id, 'mode': mode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aiom-db-all-in-one-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_100_worst_movies(mode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Showing WORST movies of all the time"
    
    """
    url = f"https://aiom-db-all-in-one-movie-database.p.rapidapi.com/showtime/"
    querystring = {'mode': mode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aiom-db-all-in-one-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_100_popular_tv_series(mode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Showing POPULAR series of all the time"
    
    """
    url = f"https://aiom-db-all-in-one-movie-database.p.rapidapi.com/showtime/"
    querystring = {'mode': mode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aiom-db-all-in-one-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_movies_by_genre(genre: str, mode: str, count: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Showing TOP animation movies"
    genre: can be:
animation, history, mystery, horror, romance, comedy, crime, drama, musical, sport, family, western, war, fantasy, sci-fi, adventure, thriller, action, biography
        
    """
    url = f"https://aiom-db-all-in-one-movie-database.p.rapidapi.com/showtime/"
    querystring = {'genre': genre, 'mode': mode, }
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aiom-db-all-in-one-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_250_tv_series(mode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Showing  top 250 tv series of all time"
    
    """
    url = f"https://aiom-db-all-in-one-movie-database.p.rapidapi.com/showtime/"
    querystring = {'mode': mode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aiom-db-all-in-one-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_250_movies_all_the_time(mode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Showing  Best movies of all time"
    
    """
    url = f"https://aiom-db-all-in-one-movie-database.p.rapidapi.com/showtime/"
    querystring = {'mode': mode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aiom-db-all-in-one-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_100_popular_movies(mode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Showing POPULAR movies of all the time"
    
    """
    url = f"https://aiom-db-all-in-one-movie-database.p.rapidapi.com/showtime/"
    querystring = {'mode': mode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aiom-db-all-in-one-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_movie(mode: str, string: str, count: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for a movie with a keyword"
    
    """
    url = f"https://aiom-db-all-in-one-movie-database.p.rapidapi.com/search/"
    querystring = {'mode': mode, 'string': string, }
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aiom-db-all-in-one-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def in_theaters(mode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Show movies in theaters"
    
    """
    url = f"https://aiom-db-all-in-one-movie-database.p.rapidapi.com/show/"
    querystring = {'mode': mode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aiom-db-all-in-one-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def upcoming_movies(mode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Show upcoming movies"
    
    """
    url = f"https://aiom-db-all-in-one-movie-database.p.rapidapi.com/show/"
    querystring = {'mode': mode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aiom-db-all-in-one-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def weekly_top_10_cinema(mode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Show Top 10 movies now in cinema"
    
    """
    url = f"https://aiom-db-all-in-one-movie-database.p.rapidapi.com/show/"
    querystring = {'mode': mode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aiom-db-all-in-one-movie-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

