import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def meta(type: str, period: str, request: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Requests related to meta information regarding the Jikan REST Instance. Such as the most requested endpoints for a specific period, or just status on the REST API."
    type: anime, manga, character, person, search, top, schedule, season
        period: today, weekly or monthly
        request: requests or status
        
    """
    url = f"https://jikan1.p.rapidapi.com/meta/{request}/{type}/{period}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jikan1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top(subtype: str, page: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Top items on MyAnimeList Note: subtype returns a filtered top list of a type type item. For example, the top Anime (type) movies (subtype) Note 2: subtype is only for anime and manga types. Note 3: Date properties are returned in string as they only consist of the month and year - which is not appropriate for ISO8601"
    subtype: see API details for values
        page: page
        type: anime, manga, people, characters
        
    """
    url = f"https://jikan1.p.rapidapi.com/top/{type}/{page}/{subtype}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jikan1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(type: str, q: str, rated: str=None, status: str=None, sort: str=None, order_by: str=None, genre: str=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search results for the query NOTE: MyAnimeList only processes queries with a minimum of 3 letters. However, the search function can be used without q! Check examples below for more details."
    type: anime, manga, person, or character
        q: search query
        rated: See API Details for ratings
        status: Anime or Manga Status, see details
        type: Anime or Manga Types, see details
        sort: asc or desc
        order_by: See API details
        genre: See API Details for list of genres
        page: Page number of the results
        
    """
    url = f"https://jikan1.p.rapidapi.com/search/{type}"
    querystring = {'q': q, }
    if rated:
        querystring['rated'] = rated
    if status:
        querystring['status'] = status
    if type:
        querystring['type'] = type
    if sort:
        querystring['sort'] = sort
    if order_by:
        querystring['order_by'] = order_by
    if genre:
        querystring['genre'] = genre
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jikan1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def producer(page: str, producer_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Anime by this Producer/Studio/Licensor"
    page: page
        producer_id: producer_id
        
    """
    url = f"https://jikan1.p.rapidapi.com/producer/{producer_id}/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jikan1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def person(is_id: str, request: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A single person object with all its details"
    id: ID of Person
        request: Request: pictures
        
    """
    url = f"https://jikan1.p.rapidapi.com/person/{is_id}/{request}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jikan1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def season_archive(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All the years & their respective seasons that can be parsed from MyAnimeList"
    
    """
    url = f"https://jikan1.p.rapidapi.com/season/archive"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jikan1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def character(request: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A single character object with all its details"
    request: Request: pictures
        id: ID of Character
        
    """
    url = f"https://jikan1.p.rapidapi.com/character/{is_id}/{request}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jikan1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def genre(page: str, type: str, genre_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Anime/Manga items of the genre"
    page: page
        type: anime or manga
        genre_id: genre_id
        
    """
    url = f"https://jikan1.p.rapidapi.com/genre/{type}/{genre_id}/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jikan1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def manga(request: str, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A single manga object with all its details"
    request: Requests: characters, news, pictures, stats, forum, moreinfo, reviews, recommendations, userupdates
        id: Manga ID on MyAnimeList.net
        
    """
    url = f"https://jikan1.p.rapidapi.com/manga/{is_id}/{request}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jikan1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def schedule(day: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Anime schedule of the week or specified day"
    day: monday tuesday wednesday thursday friday saturday sunday, other (v3), unknown (v3)
        
    """
    url = f"https://jikan1.p.rapidapi.com/schedule/{day}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jikan1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def anime(request: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Resource object with all it's details"
    request: Choose between: characters_staff, episodes, news, pictures, videos, stats, forum, moreinfo, reviews, recommendations, userupdates
        id: ID of Anime on MyAnimeList.net
        
    """
    url = f"https://jikan1.p.rapidapi.com/anime/{is_id}/{request}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jikan1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def season(season: str, year: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Anime of the specified season. "
    season: summer, spring, fall, or winter
        year: year
        
    """
    url = f"https://jikan1.p.rapidapi.com/season/{year}/{season}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jikan1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def club(page: str, is_id: str, request: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A single club object with all its details"
    page: page
        id: club_id
        request: request
        
    """
    url = f"https://jikan1.p.rapidapi.com/club/{is_id}/{request}/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jikan1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def magazine(page: str, magazine_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Manga by this Magazine/Serializer/Publisher"
    page: page
        magazine_id: magazine_id
        
    """
    url = f"https://jikan1.p.rapidapi.com/magazine/{magazine_id}/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jikan1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def season_later(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Anime that have been announced for the upcoming seasons"
    
    """
    url = f"https://jikan1.p.rapidapi.com/season/later"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jikan1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user(argument: str, username: str, request: str, year: str=None, sort: str=None, order_by2: str=None, aired_from: str=None, aired_to: str=None, producer: str=None, airing_status: str=None, search: str=None, order_by: str=None, season: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User related data Note: About is returned in HTML as MyAnimeList allows custom "about" sections for users that can consist of images, formatting, etc. Note 2: Anime & Manga Lists are paginated. Only 300 items are returned per page."
    argument: argument
        username: username
        request: Request: profile, history, friends, animelist, mangalist
        year: YYYY
        sort: See API Details for values
        order_by2: See API Details for values
        aired_from: yyyy-mm-dd
        aired_to: yyyy-mm-dd
        producer: producer_id
        airing_status: See API Details for values
        search: Return items in your list matching the string
        order_by: See API Details for values
        season: summer, spring, fall, or winter
        
    """
    url = f"https://jikan1.p.rapidapi.com/user/{username}/{request}/{argument}"
    querystring = {}
    if year:
        querystring['year'] = year
    if sort:
        querystring['sort'] = sort
    if order_by2:
        querystring['order_by2'] = order_by2
    if aired_from:
        querystring['aired_from'] = aired_from
    if aired_to:
        querystring['aired_to'] = aired_to
    if producer:
        querystring['producer'] = producer
    if airing_status:
        querystring['airing_status'] = airing_status
    if search:
        querystring['search'] = search
    if order_by:
        querystring['order_by'] = order_by
    if season:
        querystring['season'] = season
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jikan1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

