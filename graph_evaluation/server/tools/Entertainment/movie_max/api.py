import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_configuration(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the system wide configuration information. Some elements of the API require some knowledge of this configuration data. The purpose of this is to try and keep the actual API responses as light as possible. It is recommended you cache this data within your application and check for updates every few days. This method currently holds the data relevant to building image URLs. To build an image URL, you will need 3 pieces of data. The base_url, size and file_path. Simply combine them all and you will have a fully qualified URL. Here’s an example URL: http://image.tmdb.org/t/p/w500/8uO0gUM8aNqYLs1OsTBQiXu0fEv.jpg"
    
    """
    url = f"https://joseph-n-movie-max-v1.p.rapidapi.com/v1/configuration"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "joseph-n-movie-max-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def movie(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get movie information for a particular movie. You must provide the movie id"
    
    """
    url = f"https://joseph-n-movie-max-v1.p.rapidapi.com/v1/movies/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "joseph-n-movie-max-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_theatres(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of all currently supported theatres by the API"
    
    """
    url = f"https://joseph-n-movie-max-v1.p.rapidapi.com/v1/theatres"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "joseph-n-movie-max-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def schedules(theatre_id: str, movie_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get showtimes of a particular movie in a particular theatre. You will need to pass in the theatre's id and the movie id. The result is a list of showtimes that the movie is showing"
    
    """
    url = f"https://joseph-n-movie-max-v1.p.rapidapi.com/v1/theatres/{theatre_id}/movies/{movie_id}/schedules"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "joseph-n-movie-max-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def theatres(movie_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of theatres that a particular movie is showing in."
    
    """
    url = f"https://joseph-n-movie-max-v1.p.rapidapi.com/v1/movies/{movie_id}/theatres"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "joseph-n-movie-max-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def upcoming(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of upcoming movies across all theatres"
    
    """
    url = f"https://joseph-n-movie-max-v1.p.rapidapi.com/v1/movies/upcoming"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "joseph-n-movie-max-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def movies(theatre_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of movies showing in a particular theatre. You will need to pass in the theatre's id which you get when you get in the previous step"
    
    """
    url = f"https://joseph-n-movie-max-v1.p.rapidapi.com/v1/theatres/{theatre_id}/movies/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "joseph-n-movie-max-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_schedules(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Accessing this enpoints returns a list of dates in which movies are showing across all theatres. To get a list of movies you must use the format of the dates AS IS in this list"
    
    """
    url = f"https://joseph-n-movie-max-v1.p.rapidapi.com/v1/schedules"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "joseph-n-movie-max-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def movies_showing(date: str, theatre_id: int=2, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of theatres with respective movies that are showing on the specified date plus their showtimes on that date. The format must be in dd-mm-yyyy. Essentially the date variable should be any date returned in /schedules otherwise you will get empty results"
    theatre_id: The id of the theatre you want to filter the results with
        
    """
    url = f"https://joseph-n-movie-max-v1.p.rapidapi.com/v1/schedules/{date}/theatres"
    querystring = {}
    if theatre_id:
        querystring['theatre_id'] = theatre_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "joseph-n-movie-max-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_movies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of movies showing across all theatres"
    
    """
    url = f"https://joseph-n-movie-max-v1.p.rapidapi.com/v1/movies"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "joseph-n-movie-max-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

