import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def static_genres(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Static list of Genres (updated daily)"
    
    """
    url = f"https://unogs-unogs-v1.p.rapidapi.com/static/genres"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unogs-unogs-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_genres(netflix_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All Genres Associated with a Title"
    netflix_id: actual netflix ID for the title e.g. 70143836 for Breaking Bad
        
    """
    url = f"https://unogs-unogs-v1.p.rapidapi.com/title/genres"
    querystring = {'netflix_id': netflix_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unogs-unogs-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_images(netflix_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All Images associated with a Title"
    netflix_id: actual netflix ID for the title e.g. 70143836 for Breaking Bad
        
    """
    url = f"https://unogs-unogs-v1.p.rapidapi.com/title/images"
    querystring = {'netflix_id': netflix_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unogs-unogs-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_details(netflix_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "General Details associated with a Title"
    netflix_id: actual netflix ID for the title e.g. 70143836 for Breaking Bad
        
    """
    url = f"https://unogs-unogs-v1.p.rapidapi.com/title/details"
    querystring = {'netflix_id': netflix_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unogs-unogs-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_episodes(season_id: int, netflix_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All Episodes associated with a Title"
    season_id: actual netflix season ID e.g. 70105286 for Breaking Bad season 1
        netflix_id: actual netflix ID for the title e.g. 70143836 for Breaking Bad
        
    """
    url = f"https://unogs-unogs-v1.p.rapidapi.com/title/episodes"
    querystring = {'season_id': season_id, 'netflix_id': netflix_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unogs-unogs-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_countries(netflix_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All Country/Language information associated with a Title"
    netflix_id: actual netflix ID for the title e.g. 70143836 for Breaking Bad
        
    """
    url = f"https://unogs-unogs-v1.p.rapidapi.com/title/countries"
    querystring = {'netflix_id': netflix_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unogs-unogs-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_titles(end_rating: str=None, country_list: str=None, start_year: int=None, person: str=None, offset: int=None, order_by: str='date', limit: int=None, end_year: int=None, top250: int=None, start_rating: str=None, new_date: str=None, top250tv: int=None, title: str=None, expiring: str=None, subtitle: str=None, type: str='movie', genre_list: str=None, audio: str=None, audio_sub_andor: str=None, country_andorunique: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Parameter Notes:
		new_date will either take a single date and check for all titles added after that date or it will take 2 dates separated by a comma to check titles added between them."
    end_rating: number between 1 and 10, aligns with imdb rating, if there is one
        country_list: CSV list of country IDs e.g. 46,78
        start_year: year date > 1900
        person: name of person in title
        offset: where you want the response to start e.g. 10 starts with the 10th element
        order_by: Order of Response
        limit: limit of object returned
        end_year: year date > 1900
        top250: IMDB top 250 Movies number 0-250
        start_rating: number between 1 and 10, aligns with imdb rating, if there is one
        new_date: 2018-01-01 or 2010-01-01,2012-01-01 see Parameter Notes
        top250tv: IMDB top 250 Series number 0-250
        title: search all titles by name
        expiring: if 'true' show all titles set to expire
        subtitle: name of subtitle language e.g. english
        type: movie or series
        genre_list: CSV list of genres, see misc/genre endpoint
        audio: name of audio language e.g english
        audio_sub_andor: and=language in bot audio and sub, or=language in either audio or sub
        country_andorunique: and=titles in all countries, or=any title in these countries, unique=unique titles in chosen countries
        
    """
    url = f"https://unogs-unogs-v1.p.rapidapi.com/search/titles"
    querystring = {}
    if end_rating:
        querystring['end_rating'] = end_rating
    if country_list:
        querystring['country_list'] = country_list
    if start_year:
        querystring['start_year'] = start_year
    if person:
        querystring['person'] = person
    if offset:
        querystring['offset'] = offset
    if order_by:
        querystring['order_by'] = order_by
    if limit:
        querystring['limit'] = limit
    if end_year:
        querystring['end_year'] = end_year
    if top250:
        querystring['top250'] = top250
    if start_rating:
        querystring['start_rating'] = start_rating
    if new_date:
        querystring['new_date'] = new_date
    if top250tv:
        querystring['top250tv'] = top250tv
    if title:
        querystring['title'] = title
    if expiring:
        querystring['expiring'] = expiring
    if subtitle:
        querystring['subtitle'] = subtitle
    if type:
        querystring['type'] = type
    if genre_list:
        querystring['genre_list'] = genre_list
    if audio:
        querystring['audio'] = audio
    if audio_sub_andor:
        querystring['audio_sub_andor'] = audio_sub_andor
    if country_andorunique:
        querystring['country_andorunique'] = country_andorunique
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unogs-unogs-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_people(netflix_id: int=None, person_type: str='Actor', limit: int=None, name: str=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for people by title name, person name, or netflix_id"
    netflix_id: actual netflix ID for the title e.g. 70143836 for Breaking Bad
        person_type: what the person did on the project
        limit: limit of object returned
        name: search all titles by name
        offset: where you want the response to start e.g. 10 starts with the 10th element
        
    """
    url = f"https://unogs-unogs-v1.p.rapidapi.com/search/people"
    querystring = {}
    if netflix_id:
        querystring['netflix_id'] = netflix_id
    if person_type:
        querystring['person_type'] = person_type
    if limit:
        querystring['limit'] = limit
    if name:
        querystring['name'] = name
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unogs-unogs-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_deleted(title: str=None, delete_date: str=None, netflix_id: int=None, limit: int=None, offset: int=None, country_list: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for Deleted Titles by Country or Title Name
		delete_date will either take a single date and check for all titles added after that date or it will take 2 dates separated by a comma to check titles added between them."
    title: search all titles by name
        delete_date: 2018-01-01 or 2010-01-01,2012-01-01 see Parameter Notes
        netflix_id: actual netflix ID for the title e.g. 70143836 for Breaking Bad
        limit: limit of object returned
        offset: where you want the response to start e.g. 10 starts with the 10th element
        country_list: CSV list of country IDs e.g. 46,78
        
    """
    url = f"https://unogs-unogs-v1.p.rapidapi.com/search/deleted"
    querystring = {}
    if title:
        querystring['title'] = title
    if delete_date:
        querystring['delete_date'] = delete_date
    if netflix_id:
        querystring['netflix_id'] = netflix_id
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if country_list:
        querystring['country_list'] = country_list
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unogs-unogs-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def static_countries(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Static list of Supported Countries"
    
    """
    url = f"https://unogs-unogs-v1.p.rapidapi.com/static/countries"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unogs-unogs-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

