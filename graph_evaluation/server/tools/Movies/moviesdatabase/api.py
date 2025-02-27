import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def titles_utils_genres(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://moviesdatabase.p.rapidapi.com/titles/utils/genres"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moviesdatabase.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def titles_utils_lists(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://moviesdatabase.p.rapidapi.com/titles/utils/lists"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moviesdatabase.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def titles_search_akas_aka(aka: str, startyear: int=None, info: str=None, page: str=None, year: int=None, sort: str=None, endyear: int=None, limit: int=None, titletype: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    aka: Title of  the Movie/Series
        startyear: Year range filter -from- ex: ?startYear=2020
        info: Info type structure (default: mini-info) -> base_info / mini_info / image /...
        page: Page number
        year: Year filter ex: ?year=2020
        sort: Add sorting (incr, decr) -> year.incr /year.decr
        endyear: Year range filter -to- ex: ?endYear=2020
        limit: Number of titles per page (default: 10) -> 10 max 
        titletype: Filter by type of title
        
    """
    url = f"https://moviesdatabase.p.rapidapi.com/titles/search/akas/{aka}"
    querystring = {}
    if startyear:
        querystring['startYear'] = startyear
    if info:
        querystring['info'] = info
    if page:
        querystring['page'] = page
    if year:
        querystring['year'] = year
    if sort:
        querystring['sort'] = sort
    if endyear:
        querystring['endYear'] = endyear
    if limit:
        querystring['limit'] = limit
    if titletype:
        querystring['titleType'] = titletype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moviesdatabase.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def titles_search_keyword_keyword(keyword: str, info: str=None, year: int=None, page: str=None, startyear: int=None, sort: str=None, titletype: str=None, limit: int=None, endyear: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    keyword: Keyword
        info: Info type structure (default: mini-info) -> base_info / mini_info / image / ...
        year: Year filter ex: ?year=2020
        page: Page number
        startyear: Year range filter -from- ex: ?startYear=2020
        sort: Add sorting (incr, decr) -> year.incr /year.decr
        titletype: Filter by type of title
        limit: Number of titles per page (default: 10) -> 10 max 
        endyear: Year range filter -to- ex: ?endYear=2020
        
    """
    url = f"https://moviesdatabase.p.rapidapi.com/titles/search/keyword/{keyword}"
    querystring = {}
    if info:
        querystring['info'] = info
    if year:
        querystring['year'] = year
    if page:
        querystring['page'] = page
    if startyear:
        querystring['startYear'] = startyear
    if sort:
        querystring['sort'] = sort
    if titletype:
        querystring['titleType'] = titletype
    if limit:
        querystring['limit'] = limit
    if endyear:
        querystring['endYear'] = endyear
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moviesdatabase.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def titles_search_title_title(title: str, exact: bool=None, info: str=None, year: int=None, page: str=None, sort: str=None, endyear: int=None, startyear: int=None, titletype: str='movie', limit: int=None, list: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search by Title"
    title: Title of  the Movie/Series
        exact: Search by exact title
        info: Info type structure (default: mini-info) -> base_info / mini_info / image /...
        year: Year filter ex: ?year=2020
        page: Page number
        sort: Add sorting (incr, decr) -> year.incr /year.decr
        endyear: Year range filter -to- ex: ?endYear=2020
        startyear: Year range filter -from- ex: ?startYear=2020
        titletype: Filter by type of title
        limit: Number of titles per page (default: 10) -> 10 max 
        list: Selected list -> most_pop_movies / most_pop_series / top_rated_series_250 / ...
        
    """
    url = f"https://moviesdatabase.p.rapidapi.com/titles/search/title/{title}"
    querystring = {}
    if exact:
        querystring['exact'] = exact
    if info:
        querystring['info'] = info
    if year:
        querystring['year'] = year
    if page:
        querystring['page'] = page
    if sort:
        querystring['sort'] = sort
    if endyear:
        querystring['endYear'] = endyear
    if startyear:
        querystring['startYear'] = startyear
    if titletype:
        querystring['titleType'] = titletype
    if limit:
        querystring['limit'] = limit
    if list:
        querystring['list'] = list
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moviesdatabase.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def actors_random(limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    limit: Number of titles per page (default: 10) -> 10 max 
        
    """
    url = f"https://moviesdatabase.p.rapidapi.com/actors/random"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moviesdatabase.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def actors_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    id: Actor imdb id
        
    """
    url = f"https://moviesdatabase.p.rapidapi.com/actors/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moviesdatabase.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def titles_utils_titletypes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://moviesdatabase.p.rapidapi.com/titles/utils/titleTypes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moviesdatabase.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def actors(page: str=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    page: Page number
        limit: Number of titles per page (default: 10) -> 10 max 
        
    """
    url = f"https://moviesdatabase.p.rapidapi.com/actors"
    querystring = {}
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moviesdatabase.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def titles_series_seriesid(seriesid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    seriesid: Series Imdb Id
        
    """
    url = f"https://moviesdatabase.p.rapidapi.com/titles/series/{seriesid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moviesdatabase.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def titles_x_upcoming(titletype: str=None, year: int=None, genre: str=None, info: str=None, endyear: int=None, limit: int=None, sort: str=None, page: str=None, startyear: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    titletype: Filter by type of title
        year: Year filter ex: ?year=2022
        genre: Year filter ex: ?genre=Drama
        info: Info type structure (default: mini-info) -> base_info / mini_info / image / ...
        endyear: Year range filter -to- ex: ?endYear=2020
        limit: Number of titles per page (default: 10) -> 10 max 
        sort: Add sorting (incr, decr) -> year.incr / year.decr
        page: Page number
        startyear: Year range filter -from- ex: ?startYear=2020
        
    """
    url = f"https://moviesdatabase.p.rapidapi.com/titles/x/upcoming"
    querystring = {}
    if titletype:
        querystring['titleType'] = titletype
    if year:
        querystring['year'] = year
    if genre:
        querystring['genre'] = genre
    if info:
        querystring['info'] = info
    if endyear:
        querystring['endYear'] = endyear
    if limit:
        querystring['limit'] = limit
    if sort:
        querystring['sort'] = sort
    if page:
        querystring['page'] = page
    if startyear:
        querystring['startYear'] = startyear
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moviesdatabase.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def titles_id_ratings(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    id: Imdb Id of title ex: tt0000002
        
    """
    url = f"https://moviesdatabase.p.rapidapi.com/titles/{is_id}/ratings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moviesdatabase.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def titles_id_crew(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    id: Imdb Id of title ex: tt0000002
        
    """
    url = f"https://moviesdatabase.p.rapidapi.com/titles/{is_id}/crew"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moviesdatabase.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def titles_series_seriesid_season(season: str, seriesid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    season: Season number
        seriesid: Series Imdb Id
        
    """
    url = f"https://moviesdatabase.p.rapidapi.com/titles/series/{seriesid}/{season}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moviesdatabase.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def titles_episode_id(is_id: str, info: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    id: Episode Imdb Id
        info: Info type structure (default: mini-info) -> base_info / mini_info / image / ...
        
    """
    url = f"https://moviesdatabase.p.rapidapi.com/titles/episode/{is_id}"
    querystring = {}
    if info:
        querystring['info'] = info
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moviesdatabase.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def titles_id_main_actors(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    id: Imdb Id of title ex: tt0000002
        
    """
    url = f"https://moviesdatabase.p.rapidapi.com/titles/{is_id}/main_actors"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moviesdatabase.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def titles_x_titles_by_ids(idslist: str, list: str=None, info: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Titles by ids list"
    idslist: Imdb id's comma separated -> tt0001702,tt0001856,tt0001856 ...
        list: Selected list -> most_pop_movies / most_pop_series / top_rated_series_250 / ...
        info: Info type structure (default: mini-info) -> base_info / mini_info / image / ...
        
    """
    url = f"https://moviesdatabase.p.rapidapi.com/titles/x/titles-by-ids"
    querystring = {'idsList': idslist, }
    if list:
        querystring['list'] = list
    if info:
        querystring['info'] = info
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moviesdatabase.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def titles_id_aka(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    id: Imdb Id of title ex: tt0000002
        
    """
    url = f"https://moviesdatabase.p.rapidapi.com/titles/{is_id}/aka"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moviesdatabase.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def titles_random(startyear: int=None, genre: str=None, titletype: str=None, sort: str=None, limit: int=None, info: str=None, endyear: int=None, year: int=None, list: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    startyear: Year range filter -from- ex: ?startYear=2020
        genre: Year filter ex: ?genre=Drama
        titletype: Filter by type of title
        sort: Add sorting (incr, decr) -> year.incr /year.decr
        limit: Number of titles per page (default: 10) -> 10 max 
        info: Info type structure (default: mini-info) -> base_info / mini_info / image / ...
        endyear: Year range filter -to- ex: ?endYear=2020
        year: Year filter ex: ?year=2020
        list: Selected list -> most_pop_movies / most_pop_series / top_rated_series_250 / ...
        
    """
    url = f"https://moviesdatabase.p.rapidapi.com/titles/random"
    querystring = {}
    if startyear:
        querystring['startYear'] = startyear
    if genre:
        querystring['genre'] = genre
    if titletype:
        querystring['titleType'] = titletype
    if sort:
        querystring['sort'] = sort
    if limit:
        querystring['limit'] = limit
    if info:
        querystring['info'] = info
    if endyear:
        querystring['endYear'] = endyear
    if year:
        querystring['year'] = year
    if list:
        querystring['list'] = list
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moviesdatabase.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def titles_id(is_id: str, info: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    id: Imdb Id of title ex: tt0000002
        info: Info type structure (default: base-info) -> base_info / mini_info / image / ...
        
    """
    url = f"https://moviesdatabase.p.rapidapi.com/titles/{is_id}"
    querystring = {}
    if info:
        querystring['info'] = info
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moviesdatabase.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def titles(genre: str=None, startyear: int=None, titletype: str=None, list: str=None, year: int=None, sort: str=None, page: str=None, info: str=None, endyear: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    genre: Year filter ex: ?genre=Drama
        startyear: Year range filter -from- ex: ?startYear=2020
        titletype: Filter by type of title
        list: Selected list -> most_pop_movies / most_pop_series / top_rated_series_250 / ...
        year: Year filter ex: ?year=2020
        sort: Add sorting (incr, decr) -> year.incr /year.decr
        page: Page number
        info: Info type structure (default: mini-info) -> base_info / mini_info / image / ...
        endyear: Year range filter -to- ex: ?endYear=2020
        limit: Number of titles per page (default: 10) -> 10 max 
        
    """
    url = f"https://moviesdatabase.p.rapidapi.com/titles"
    querystring = {}
    if genre:
        querystring['genre'] = genre
    if startyear:
        querystring['startYear'] = startyear
    if titletype:
        querystring['titleType'] = titletype
    if list:
        querystring['list'] = list
    if year:
        querystring['year'] = year
    if sort:
        querystring['sort'] = sort
    if page:
        querystring['page'] = page
    if info:
        querystring['info'] = info
    if endyear:
        querystring['endYear'] = endyear
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moviesdatabase.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def titles_seasons_seriesid(seriesid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    seriesid: Series Imdb Id
        
    """
    url = f"https://moviesdatabase.p.rapidapi.com/titles/seasons/{seriesid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moviesdatabase.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

