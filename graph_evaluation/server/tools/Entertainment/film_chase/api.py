import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getfilms(items: int=20, include_outdated: bool=None, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a paginated list of films currently showing in cinemas, with an optional parameter to include films that are no longer showing."
    items: Number of items to return per page. Maximum value: 100
        include_outdated: Include films that are no longer showing in cinemas.
        page: Page offset to fetch.
        
    """
    url = f"https://film-chase1.p.rapidapi.com/films"
    querystring = {}
    if items:
        querystring['items'] = items
    if include_outdated:
        querystring['include_outdated'] = include_outdated
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "film-chase1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getfilmsid(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a film."
    id: Film ID.
        
    """
    url = f"https://film-chase1.p.rapidapi.com/films/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "film-chase1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getshowtimesid(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return a showtime."
    id: Showtime ID.
        
    """
    url = f"https://film-chase1.p.rapidapi.com/showtimes/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "film-chase1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcinemasid(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return a cinema."
    id: Cinema ID.
        
    """
    url = f"https://film-chase1.p.rapidapi.com/cinemas/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "film-chase1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getshowtimes(cinema_id: int=None, items: int=20, page: int=1, film_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a paginated list of showtimes."
    cinema_id: Filter by a specific cinema.
        items: Number of items to return per page. Maximum value: 100
        page: Page offset to fetch.
        film_id: Filter by a specific film.
        
    """
    url = f"https://film-chase1.p.rapidapi.com/showtimes"
    querystring = {}
    if cinema_id:
        querystring['cinema_id'] = cinema_id
    if items:
        querystring['items'] = items
    if page:
        querystring['page'] = page
    if film_id:
        querystring['film_id'] = film_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "film-chase1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcinemas(items: int=20, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a paginated list of cinemas."
    items: Number of items to return per page. Maximum value: 100
        page: Page offset to fetch.
        
    """
    url = f"https://film-chase1.p.rapidapi.com/cinemas"
    querystring = {}
    if items:
        querystring['items'] = items
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "film-chase1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

