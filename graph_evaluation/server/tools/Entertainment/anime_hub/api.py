import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_the_full_search(month: str=None, type: str=None, status: str=None, day: str=None, scores: str=None, rating: str=None, has: int=None, maxresults: int=None, term: str='Sakura', producer: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the Full Search
		
		**Parameters**
		- maxResults       // how many results at most (default: 50)
		- has     // If you already have results and just want what follows it, you can say it here. Allows pagination!
		- term     // search term
		- type        // 0-> none, else go check get helper endpoint
		- status     // 0-> none, else go check get helper endpoint
		- producer     // 0-> none, else go check get helper endpoint
		- scores        // 0-> none, else go check get helper endpoint
		- rating       // 0-> none, else go check get helper endpoint
		- day
		- month
		- year
		- genreType      // 0 for include genre list, 1 for exclude genre list
		- genres        // Go check get helper endpoint"
    
    """
    url = f"https://anime-hub.p.rapidapi.com/searchfull"
    querystring = {}
    if month:
        querystring['month'] = month
    if type:
        querystring['type'] = type
    if status:
        querystring['status'] = status
    if day:
        querystring['day'] = day
    if scores:
        querystring['scores'] = scores
    if rating:
        querystring['rating'] = rating
    if has:
        querystring['has'] = has
    if maxresults:
        querystring['maxResults'] = maxresults
    if term:
        querystring['term'] = term
    if producer:
        querystring['producer'] = producer
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "anime-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_anime_helpers(help: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is useful to get Full search parameters...
		
		**Parameters:**
		-> help (it can either be: **type, score, producer, rating, genres, genreType** )"
    
    """
    url = f"https://anime-hub.p.rapidapi.com/gethelpers"
    querystring = {'help': help, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "anime-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_anime_pictures(name: str, is_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns Pictures from a particular Anime.
		
		**Parameters:**
		name
		id"
    
    """
    url = f"https://anime-hub.p.rapidapi.com/getpictures"
    querystring = {'name': name, }
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "anime-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_anime_stats(name: str, is_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns anime stats:
		
		**Parameters**
		name
		id"
    
    """
    url = f"https://anime-hub.p.rapidapi.com/getstats"
    querystring = {'name': name, }
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "anime-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_anime_recommendation_list(name: str, is_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a list of recommendation
		
		**Parameters:**
		name
		id (The unique identifier of this anime)"
    
    """
    url = f"https://anime-hub.p.rapidapi.com/reclist"
    querystring = {'name': name, }
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "anime-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_anime_list_reviews(name: str, skip: int=20, limit: int=3, is_id: int=20047, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Endpoint Returns a list Review
		
		**Parameters:**
		name
		id
		limit [optionnal] The number max of reviews to fetch - can be really long if omit
		skip [optionnal] The number of reviews to skip"
    
    """
    url = f"https://anime-hub.p.rapidapi.com/reviewslist"
    querystring = {'name': name, }
    if skip:
        querystring['skip'] = skip
    if limit:
        querystring['limit'] = limit
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "anime-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_anime_episode_list(name: str, is_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a List of Edpisodes
		
		**Parameters:**
		name
		id"
    
    """
    url = f"https://anime-hub.p.rapidapi.com/episodelist"
    querystring = {'name': name, }
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "anime-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_anime_news(nnews: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the News of a Particular Anime
		
		**Parameters:**
		-nNews"
    
    """
    url = f"https://anime-hub.p.rapidapi.com/getnews"
    querystring = {'nNews': nnews, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "anime-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_anime_watchlist_from_user(after: str, username: str, type: str='anime', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns anime watchlist from a Particular User.
		
		**Parameters:**
		- username
		- after ((Useful to paginate. Is the number of results you want to start from. By default, MAL returns 300 entries only.)
		- type (Optional, can be either anime or manga)"
    
    """
    url = f"https://anime-hub.p.rapidapi.com/watchlistuser"
    querystring = {'after': after, 'username': username, }
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "anime-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_anime_season(season: str, year: int, type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint get the list of anime, OVAs, movies and ONAs released (or planned to be released) during the season of the specified year.
		
		**Parameters:**
		- year
		- season
		- type (The type, must be either TV, TVNew, TVCon, ONAs, OVAs, Specials or Movies)"
    
    """
    url = f"https://anime-hub.p.rapidapi.com/getseason"
    querystring = {'season': season, 'year': year, }
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "anime-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_anime_result_from_search(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a object from search result.
		
		**Parameters:**"
    
    """
    url = f"https://anime-hub.p.rapidapi.com/searchresult"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "anime-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_anime_info_from_url(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Endpoint Returns Anime Info From URL.
		
		**Parameters:**
		- url (The URL to the anime)"
    
    """
    url = f"https://anime-hub.p.rapidapi.com/getinfourl"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "anime-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_anime_information(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Endpoint returns a Anime Information.
		
		**Parameters**
		- Name (The name of the anime to search, the best match corresponding to that name will be returned).
		-"
    
    """
    url = f"https://anime-hub.p.rapidapi.com/getinfo"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "anime-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

