import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def movie_details(movie_id: int=10, imdb_id: int=10, with_cast: bool=None, with_images: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the information about a specific movie"
    movie_id: The YTS ID of the movie or the IMDB ID
        imdb_id: The YTS ID of the movie or the IMDB ID
        with_cast: When set the data returned will include the added information about the cast
        with_images: When set the data returned will include the added image URLs
        
    """
    url = f"https://list-movies-v3.p.rapidapi.com/api/v2/movie_details.json"
    querystring = {}
    if movie_id:
        querystring['movie_id'] = movie_id
    if imdb_id:
        querystring['imdb_id'] = imdb_id
    if with_cast:
        querystring['with_cast'] = with_cast
    if with_images:
        querystring['with_images'] = with_images
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "list-movies-v3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def movie_parental_guides(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all the parental guide ratings for the specified movie"
    
    """
    url = f"https://list-movies-v3.p.rapidapi.com/api/v2/movie_parental_guides.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "list-movies-v3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def movie_suggestions(movie_id: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns 4 related movies as suggestions for the user"
    
    """
    url = f"https://list-movies-v3.p.rapidapi.com/api/v2/movie_suggestions.json"
    querystring = {}
    if movie_id:
        querystring['movie_id'] = movie_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "list-movies-v3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_movies(genre: str='all', order_by: str='desc', with_rt_ratings: bool=None, sort_by: str='date_added', page: int=1, query_term: str='0', quality: str='all', minimum_rating: int=0, limit: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Used to list and search through out all the available movies. Can sort, filter, search and order the results"
    genre: Used to filter by a given genre (See http://www.imdb.com/genre/ for full list)
        order_by: Orders the results by either Ascending or Descending order.
        with_rt_ratings: Returns the list with the Rotten Tomatoes rating included.
        sort_by: Sorts the results by choosen value.
        page: used to see the next page of movies, eg limit=15 and page=2 will show you movies 15-30.
        query_term: Used for movie search, matching on: Movie Title/IMDB Code, Actor Name/IMDb, Director Name/IMDb Code.
        quality: Used to filter by a given quality.
        minimum_rating: used to filter movie by a given minimum IMDB rating.
        limit: The limit of results per page that has been set.
        
    """
    url = f"https://list-movies-v3.p.rapidapi.com/api/v2/list_movies.json"
    querystring = {}
    if genre:
        querystring['genre'] = genre
    if order_by:
        querystring['order_by'] = order_by
    if with_rt_ratings:
        querystring['with_rt_ratings'] = with_rt_ratings
    if sort_by:
        querystring['sort_by'] = sort_by
    if page:
        querystring['page'] = page
    if query_term:
        querystring['query_term'] = query_term
    if quality:
        querystring['quality'] = quality
    if minimum_rating:
        querystring['minimum_rating'] = minimum_rating
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "list-movies-v3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

