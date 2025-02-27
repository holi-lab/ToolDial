import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_languages(offset: int=None, language: str=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for language by language name used by audio and subtitle searches"
    offset: where you want the response to start e.g. 10 starts with the 10th element
        language: search language by language name
        limit: limit of object returned
        
    """
    url = f"https://unogs-disney.p.rapidapi.com/search/languages"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if language:
        querystring['language'] = language
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unogs-disney.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_titles(country_andorunique: str=None, title: str=None, start_rating: str=None, type: str='movie', order_by: str='date', subtitle_id: str=None, genre_id: str=None, limit: int=None, end_rating: str=None, audio_sub_andor: str=None, start_year: int=None, top250: int=None, top250tv: int=None, offset: int=None, new_date: str=None, end_year: int=None, person_id: str=None, audio_id: str=None, country_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Parameter Notes:
		new_date will either take a single date and check for all titles added after that date or it will take 2 dates separated by a comma to check titles added between them."
    country_andorunique: and=titles in all countries, or=any title in these countries, unique=unique titles in chosen countries
        title: search all titles by name
        start_rating: number between 1 and 10, aligns with imdb rating, if there is one
        type: movie or series
        order_by: Order of Response
        subtitle_id: search by subtitle with CSV list of language_ids see /search/language
        genre_id: search by genre with CSV list of genres, see /search/genre
        limit: limit of object returned
        end_rating: number between 1 and 10, aligns with imdb rating, if there is one
        audio_sub_andor: and=language in bot audio and sub, or=language in either audio or sub
        start_year: year date > 1900
        top250: IMDB top 250 Movies number 0-250
        top250tv: IMDB top 250 Series number 0-250
        offset: where you want the response to start e.g. 10 starts with the 10th element
        new_date: 2018-01-01 or 2010-01-01,2012-01-01 see Parameter Notes
        end_year: year date > 1900
        person_id: search by person with CSV list of people, see /search/people
        audio_id: search by audio with CSV list of language_ids see /search/language
        country_id: CSV list of country IDs e.g. 46,78
        
    """
    url = f"https://unogs-disney.p.rapidapi.com/search/titles"
    querystring = {}
    if country_andorunique:
        querystring['country_andorunique'] = country_andorunique
    if title:
        querystring['title'] = title
    if start_rating:
        querystring['start_rating'] = start_rating
    if type:
        querystring['type'] = type
    if order_by:
        querystring['order_by'] = order_by
    if subtitle_id:
        querystring['subtitle_id'] = subtitle_id
    if genre_id:
        querystring['genre_id'] = genre_id
    if limit:
        querystring['limit'] = limit
    if end_rating:
        querystring['end_rating'] = end_rating
    if audio_sub_andor:
        querystring['audio_sub_andor'] = audio_sub_andor
    if start_year:
        querystring['start_year'] = start_year
    if top250:
        querystring['top250'] = top250
    if top250tv:
        querystring['top250tv'] = top250tv
    if offset:
        querystring['offset'] = offset
    if new_date:
        querystring['new_date'] = new_date
    if end_year:
        querystring['end_year'] = end_year
    if person_id:
        querystring['person_id'] = person_id
    if audio_id:
        querystring['audio_id'] = audio_id
    if country_id:
        querystring['country_id'] = country_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unogs-disney.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_deleted(offset: int=None, country_code: str=None, limit: int=None, title_id: int=None, delete_date: str=None, country_id: str=None, title: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for Deleted Titles by Country or Title Name
		delete_date will either take a single date and check for all titles added after that date or it will take 2 dates separated by a comma to check titles added between them."
    offset: where you want the response to start e.g. 10 starts with the 10th element
        country_code: 2 letter country code e.g. GB
        limit: limit of object returned
        title_id: actual netflix ID for the title e.g. 3jLIGMDYINqD for The Mandalorian
        delete_date: 2018-01-01 or 2010-01-01,2012-01-01 see Parameter Notes
        country_id: CSV list of country IDs e.g. 46,78
        title: search all titles by name
        
    """
    url = f"https://unogs-disney.p.rapidapi.com/search/deleted"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if country_code:
        querystring['country_code'] = country_code
    if limit:
        querystring['limit'] = limit
    if title_id:
        querystring['title_id'] = title_id
    if delete_date:
        querystring['delete_date'] = delete_date
    if country_id:
        querystring['country_id'] = country_id
    if title:
        querystring['title'] = title
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unogs-disney.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_genres(limit: int=None, name: str=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for genre by genre name"
    limit: limit of object returned
        name: search for all queries by query name
        offset: where you want the response to start e.g. 10 starts with the 10th element
        
    """
    url = f"https://unogs-disney.p.rapidapi.com/search/genres"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if name:
        querystring['name'] = name
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unogs-disney.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_countries(country: str=None, limit: int=None, country_code: str=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for all available countries by name"
    country: search country by country name
        limit: limit of object returned
        country_code: search by 2 letter country code e.g. GB for Great Britain
        offset: where you want the response to start e.g. 10 starts with the 10th element
        
    """
    url = f"https://unogs-disney.p.rapidapi.com/search/countries"
    querystring = {}
    if country:
        querystring['country'] = country
    if limit:
        querystring['limit'] = limit
    if country_code:
        querystring['country_code'] = country_code
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unogs-disney.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_people(limit: int=None, offset: int=None, name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for people by person name"
    limit: limit of object returned
        offset: where you want the response to start e.g. 10 starts with the 10th element
        name: search for a person by their name e.g. cruise for Tom Cruise
        
    """
    url = f"https://unogs-disney.p.rapidapi.com/search/people"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unogs-disney.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_images(title_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All Images associated with a Title"
    title_id: actual title ID for the title e.g. 70143836 for Breaking Bad
        
    """
    url = f"https://unogs-disney.p.rapidapi.com/title/images"
    querystring = {'title_id': title_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unogs-disney.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_genres(title_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All Genres Associated with a Title"
    title_id: actual title ID for the title e.g. 3jLIGMDYINqD for The Mandalorian
        
    """
    url = f"https://unogs-disney.p.rapidapi.com/title/genres"
    querystring = {'title_id': title_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unogs-disney.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_people(title_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All People Associated with a Title"
    title_id: actual title ID for the title e.g. 3jLIGMDYINqD for The Mandalorian
        
    """
    url = f"https://unogs-disney.p.rapidapi.com/title/people"
    querystring = {'title_id': title_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unogs-disney.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_episodes(title_id: int, season_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All Episodes associated with a Title"
    title_id: actual title ID for the title e.g. 3jLIGMDYINqD for The Mandalorian
        season_id: actual title season ID e.g. b423b024-b641-45e8-a075-575c35292100 for Mandalorian season 1
        
    """
    url = f"https://unogs-disney.p.rapidapi.com/title/episodes"
    querystring = {'title_id': title_id, 'season_id': season_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unogs-disney.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_countries(title_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All Country/Language information associated with a Title"
    title_id: actual title ID for the title e.g. 70143836 for Breaking Bad
        
    """
    url = f"https://unogs-disney.p.rapidapi.com/title/countries"
    querystring = {'title_id': title_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unogs-disney.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_details(title_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "General Details associated with a Title"
    title_id: actual title ID for the title e.g. 3jLIGMDYINqD for The Mandalorian
        
    """
    url = f"https://unogs-disney.p.rapidapi.com/title/details"
    querystring = {'title_id': title_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unogs-disney.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

