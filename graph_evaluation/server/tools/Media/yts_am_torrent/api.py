import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def list_movies_xml(genre: str=None, limit: int=None, query_term: str=None, quality: str=None, minimum_rating: int=None, with_rt_ratings: bool=None, page: int=None, sort_by: str=None, order_by: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Used to list and search through out all the available movies. Can sort, filter, search and order the results"
    genre: 	Used to filter by a given genre (See http://www.imdb.com/genre/ for full list)
        limit: Integer between 1 - 50 (inclusive): The limit of results per page that has been set
        query_term: Used for movie search, matching on: Movie Title/IMDb Code, Actor Name/IMDb Code, Director Name/IMDb Code
        quality: String (720p, 1080p, 3D): 	Used to filter by a given quality
        minimum_rating: 	Integer between 0 - 9 (inclusive): 	Used to filter movie by a given minimum IMDb rating
        with_rt_ratings: Returns the list with the Rotten Tomatoes rating included
        page: Integer (Unsigned): Used to see the next page of movies, eg limit=15 and page=2 will show you movies 15-30
        sort_by: String (title, year, rating, peers, seeds, download_count, like_count, date_added): Sorts the results by choosen value
        order_by: String (desc, asc): Orders the results by either Ascending or Descending order
        
    """
    url = f"https://yts-am-torrent.p.rapidapi.com/list_movies.jsonp"
    querystring = {}
    if genre:
        querystring['genre'] = genre
    if limit:
        querystring['limit'] = limit
    if query_term:
        querystring['query_term'] = query_term
    if quality:
        querystring['quality'] = quality
    if minimum_rating:
        querystring['minimum_rating'] = minimum_rating
    if with_rt_ratings:
        querystring['with_rt_ratings'] = with_rt_ratings
    if page:
        querystring['page'] = page
    if sort_by:
        querystring['sort_by'] = sort_by
    if order_by:
        querystring['order_by'] = order_by
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yts-am-torrent.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def movie_reviews_xml(movie_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all the IMDb movie reviews for the specified movie"
    movie_id: 	The ID of the movie
        
    """
    url = f"https://yts-am-torrent.p.rapidapi.com/movie_reviews.xml"
    querystring = {}
    if movie_id:
        querystring['movie_id'] = movie_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yts-am-torrent.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def movie_details_xml(movie_id: int, with_cast: bool=None, with_images: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the information about a specific movie"
    movie_id: The ID of the movie
        with_cast: When set the data returned will include the added information about the cast
        with_images: When set the data returned will include the added image URLs
        
    """
    url = f"https://yts-am-torrent.p.rapidapi.com/movie_details.xml"
    querystring = {'movie_id': movie_id, }
    if with_cast:
        querystring['with_cast'] = with_cast
    if with_images:
        querystring['with_images'] = with_images
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yts-am-torrent.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def movie_suggestions_xml(movie_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns 4 related movies as suggestions for the user"
    movie_id: The ID of the movie
        
    """
    url = f"https://yts-am-torrent.p.rapidapi.com/movie_suggestions.xml"
    querystring = {}
    if movie_id:
        querystring['movie_id'] = movie_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yts-am-torrent.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def movie_details_jsonp(movie_id: int, with_cast: bool=None, with_images: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the information about a specific movie"
    movie_id: The ID of the movie
        with_cast: When set the data returned will include the added information about the cast
        with_images: When set the data returned will include the added image URLs
        
    """
    url = f"https://yts-am-torrent.p.rapidapi.com/movie_details.jsonp"
    querystring = {'movie_id': movie_id, }
    if with_cast:
        querystring['with_cast'] = with_cast
    if with_images:
        querystring['with_images'] = with_images
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yts-am-torrent.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_details_json(user_id: int, with_recently_downloaded: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the user details"
    user_id: The ID of the user
        with_recently_downloaded: If set it will add the most recent downloads by the specified user
        
    """
    url = f"https://yts-am-torrent.p.rapidapi.com/user_details.json"
    querystring = {'user_id': user_id, }
    if with_recently_downloaded:
        querystring['with_recently_downloaded'] = with_recently_downloaded
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yts-am-torrent.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_movies_jsonp(limit: int=None, sort_by: str=None, page: int=None, genre: str=None, quality: str=None, with_rt_ratings: bool=None, minimum_rating: int=None, query_term: str=None, order_by: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Used to list and search through out all the available movies. Can sort, filter, search and order the results"
    limit: Integer between 1 - 50 (inclusive): The limit of results per page that has been set
        sort_by: String (title, year, rating, peers, seeds, download_count, like_count, date_added): Sorts the results by choosen value
        page: Integer (Unsigned): Used to see the next page of movies, eg limit=15 and page=2 will show you movies 15-30
        genre: 	Used to filter by a given genre (See http://www.imdb.com/genre/ for full list)
        quality: String (720p, 1080p, 3D): 	Used to filter by a given quality
        with_rt_ratings: Returns the list with the Rotten Tomatoes rating included
        minimum_rating: 	Integer between 0 - 9 (inclusive): 	Used to filter movie by a given minimum IMDb rating
        query_term: Used for movie search, matching on: Movie Title/IMDb Code, Actor Name/IMDb Code, Director Name/IMDb Code
        order_by: String (desc, asc): Orders the results by either Ascending or Descending order
        
    """
    url = f"https://yts-am-torrent.p.rapidapi.com/list_movies.jsonp"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if sort_by:
        querystring['sort_by'] = sort_by
    if page:
        querystring['page'] = page
    if genre:
        querystring['genre'] = genre
    if quality:
        querystring['quality'] = quality
    if with_rt_ratings:
        querystring['with_rt_ratings'] = with_rt_ratings
    if minimum_rating:
        querystring['minimum_rating'] = minimum_rating
    if query_term:
        querystring['query_term'] = query_term
    if order_by:
        querystring['order_by'] = order_by
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yts-am-torrent.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def movie_reviews_jsonp(movie_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all the IMDb movie reviews for the specified movie"
    movie_id: 	The ID of the movie
        
    """
    url = f"https://yts-am-torrent.p.rapidapi.com/movie_reviews.json"
    querystring = {}
    if movie_id:
        querystring['movie_id'] = movie_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yts-am-torrent.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_details_jsonp(user_id: int, with_recently_downloaded: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the user details"
    user_id: The ID of the user
        with_recently_downloaded: If set it will add the most recent downloads by the specified user
        
    """
    url = f"https://yts-am-torrent.p.rapidapi.com/user_details.jsonp"
    querystring = {'user_id': user_id, }
    if with_recently_downloaded:
        querystring['with_recently_downloaded'] = with_recently_downloaded
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yts-am-torrent.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def movie_suggestions_json(movie_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns 4 related movies as suggestions for the user"
    movie_id: The ID of the movie
        
    """
    url = f"https://yts-am-torrent.p.rapidapi.com/movie_suggestions.json"
    querystring = {}
    if movie_id:
        querystring['movie_id'] = movie_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yts-am-torrent.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def movie_suggestions_jsonp(movie_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns 4 related movies as suggestions for the user"
    movie_id: The ID of the movie
        
    """
    url = f"https://yts-am-torrent.p.rapidapi.com/movie_suggestions.jsonp"
    querystring = {}
    if movie_id:
        querystring['movie_id'] = movie_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yts-am-torrent.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_upcoming_xml(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the 4 latest upcoming movies"
    
    """
    url = f"https://yts-am-torrent.p.rapidapi.com/list_upcoming.xml"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yts-am-torrent.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def movie_reviews_json(movie_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all the IMDb movie reviews for the specified movie"
    movie_id: 	The ID of the movie
        
    """
    url = f"https://yts-am-torrent.p.rapidapi.com/movie_reviews.json"
    querystring = {}
    if movie_id:
        querystring['movie_id'] = movie_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yts-am-torrent.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def movie_details_json(movie_id: int, with_cast: bool=None, with_images: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the information about a specific movie"
    movie_id: The ID of the movie
        with_cast: When set the data returned will include the added information about the cast
        with_images: When set the data returned will include the added image URLs
        
    """
    url = f"https://yts-am-torrent.p.rapidapi.com/movie_details.json"
    querystring = {'movie_id': movie_id, }
    if with_cast:
        querystring['with_cast'] = with_cast
    if with_images:
        querystring['with_images'] = with_images
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yts-am-torrent.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def movie_comments_jsonp(movie_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all the comments for the specified movie"
    movie_id: The ID of the movie
        
    """
    url = f"https://yts-am-torrent.p.rapidapi.com/movie_comments.jsonp"
    querystring = {}
    if movie_id:
        querystring['movie_id'] = movie_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yts-am-torrent.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_movies_json(with_rt_ratings: bool=None, minimum_rating: int=None, limit: int=None, page: int=None, query_term: str=None, order_by: str=None, genre: str=None, quality: str=None, sort_by: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Used to list and search through out all the available movies. Can sort, filter, search and order the results"
    with_rt_ratings: Returns the list with the Rotten Tomatoes rating included
        minimum_rating: 	Integer between 0 - 9 (inclusive): 	Used to filter movie by a given minimum IMDb rating
        limit: Integer between 1 - 50 (inclusive): The limit of results per page that has been set
        page: Integer (Unsigned): Used to see the next page of movies, eg limit=15 and page=2 will show you movies 15-30
        query_term: Used for movie search, matching on: Movie Title/IMDb Code, Actor Name/IMDb Code, Director Name/IMDb Code
        order_by: String (desc, asc): Orders the results by either Ascending or Descending order
        genre: 	Used to filter by a given genre (See http://www.imdb.com/genre/ for full list)
        quality: String (720p, 1080p, 3D): 	Used to filter by a given quality
        sort_by: String (title, year, rating, peers, seeds, download_count, like_count, date_added): Sorts the results by choosen value
        
    """
    url = f"https://yts-am-torrent.p.rapidapi.com/list_movies.json"
    querystring = {}
    if with_rt_ratings:
        querystring['with_rt_ratings'] = with_rt_ratings
    if minimum_rating:
        querystring['minimum_rating'] = minimum_rating
    if limit:
        querystring['limit'] = limit
    if page:
        querystring['page'] = page
    if query_term:
        querystring['query_term'] = query_term
    if order_by:
        querystring['order_by'] = order_by
    if genre:
        querystring['genre'] = genre
    if quality:
        querystring['quality'] = quality
    if sort_by:
        querystring['sort_by'] = sort_by
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yts-am-torrent.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def movie_parental_guides_jsonp(movie_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all the parental guide ratings for the specified movie"
    movie_id: The ID of the movie
        
    """
    url = f"https://yts-am-torrent.p.rapidapi.com/movie_parental_guides.jsonp"
    querystring = {}
    if movie_id:
        querystring['movie_id'] = movie_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yts-am-torrent.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def movie_parental_guides_xml(movie_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all the parental guide ratings for the specified movie"
    movie_id: The ID of the movie
        
    """
    url = f"https://yts-am-torrent.p.rapidapi.com/movie_parental_guides.xml"
    querystring = {}
    if movie_id:
        querystring['movie_id'] = movie_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yts-am-torrent.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def movie_comments_xml(movie_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all the comments for the specified movie"
    movie_id: The ID of the movie
        
    """
    url = f"https://yts-am-torrent.p.rapidapi.com/movie_comments.jsonp"
    querystring = {}
    if movie_id:
        querystring['movie_id'] = movie_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yts-am-torrent.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_details_xml(user_id: int, with_recently_downloaded: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the user details"
    user_id: The ID of the user
        with_recently_downloaded: If set it will add the most recent downloads by the specified user
        
    """
    url = f"https://yts-am-torrent.p.rapidapi.com/user_details.xml"
    querystring = {'user_id': user_id, }
    if with_recently_downloaded:
        querystring['with_recently_downloaded'] = with_recently_downloaded
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yts-am-torrent.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def movie_parental_guides_json(movie_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all the parental guide ratings for the specified movie"
    movie_id: The ID of the movie
        
    """
    url = f"https://yts-am-torrent.p.rapidapi.com/movie_parental_guides.json"
    querystring = {}
    if movie_id:
        querystring['movie_id'] = movie_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yts-am-torrent.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def movie_comments_json(movie_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all the comments for the specified movie"
    movie_id: The ID of the movie
        
    """
    url = f"https://yts-am-torrent.p.rapidapi.com/movie_comments.json"
    querystring = {}
    if movie_id:
        querystring['movie_id'] = movie_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yts-am-torrent.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_upcoming_json(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the 4 latest upcoming movies"
    
    """
    url = f"https://yts-am-torrent.p.rapidapi.com/list_upcoming.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yts-am-torrent.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_upcoming_jsonp(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the 4 latest upcoming movies"
    
    """
    url = f"https://yts-am-torrent.p.rapidapi.com/list_upcoming.jsonp"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yts-am-torrent.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

