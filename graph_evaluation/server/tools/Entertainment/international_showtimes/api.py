import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_all_cinemas_in_a_specific_country(cinema: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of all cinemas in a particular country"
    
    """
    url = f"https://international-showtimes.p.rapidapi.com/api.internationalshowtimes.com/v4/cinemas/?countries=DE"
    querystring = {}
    if cinema:
        querystring['cinema'] = cinema
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "international-showtimes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_showtimes_per_movie_in_a_specific_city(showtimes: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all showtimes per movie in a specific City"
    
    """
    url = f"https://international-showtimes.p.rapidapi.com/api.internationalshowtimes.com/v4/showtimes?city_ids=1&movie_id=12345"
    querystring = {}
    if showtimes:
        querystring['showtimes'] = showtimes
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "international-showtimes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_movies_playing_in_a_particular_cinema(movies: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get all movies playing in a particular cinema"
    
    """
    url = f"https://international-showtimes.p.rapidapi.com/api.internationalshowtimes.com/v4/movies/?cinema_id="
    querystring = {}
    if movies:
        querystring['movies'] = movies
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "international-showtimes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

