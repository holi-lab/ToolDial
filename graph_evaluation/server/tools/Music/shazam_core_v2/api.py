import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def artist_details_deprecated(artist_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a details of artist. Ex: https://www.shazam.com/artist/the-beatles/136975"
    artist_id: Artist id
        
    """
    url = f"https://shazam-core.p.rapidapi.com/v1/artists/details"
    querystring = {'artist_id': artist_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam-core.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def multi_search(search_type: str, query: str, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Multi-search by query (song, artist). Use pagination"
    query: Query
        offset: offset
        
    """
    url = f"https://shazam-core.p.rapidapi.com/v1/search/multi"
    querystring = {'search_type': search_type, 'query': query, }
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam-core.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def chart_by_genre_and_country(genre_code: str, country_code: str, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of tracks by genre and country code. Ex: https://www.app.shazam.com/charts/genre/australia/hip-hop-rap"
    offset: Offset
        
    """
    url = f"https://shazam-core.p.rapidapi.com/v1/charts/genre-country"
    querystring = {'genre_code': genre_code, 'country_code': country_code, }
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam-core.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def track_details(track_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a details of track. Ex: https://www.shazam.com/track/216314/let-it-be"
    track_id: Track id
        
    """
    url = f"https://shazam-core.p.rapidapi.com/v1/tracks/details"
    querystring = {'track_id': track_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam-core.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def world_chart_by_genre(genre_code: str, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of tracks in word by genre. Ex: https://www.app.shazam.com/charts/top-200/world"
    offset: Offset
        
    """
    url = f"https://shazam-core.p.rapidapi.com/v1/charts/genre-world"
    querystring = {'genre_code': genre_code, }
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam-core.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def chart_by_country(country_code: str, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of tracks by country code. Ex: https://www.app.shazam.com/charts/discovery/australia"
    offset: Offset
        
    """
    url = f"https://shazam-core.p.rapidapi.com/v1/charts/country"
    querystring = {'country_code': country_code, }
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam-core.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_suggest(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search suggest drop-down list. Live search"
    query: Query
        
    """
    url = f"https://shazam-core.p.rapidapi.com/v1/search/suggest"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam-core.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def track_similarities(track_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get similarities track"
    track_id: Track id
        
    """
    url = f"https://shazam-core.p.rapidapi.com/v1/tracks/similarities"
    querystring = {'track_id': track_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam-core.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def total_shazams(track_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get total times the specific tracks is detected"
    track_id: Track id
        
    """
    url = f"https://shazam-core.p.rapidapi.com/v1/tracks/total-shazams"
    querystring = {'track_id': track_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam-core.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def schema_cities_and_genres(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of countries, cities and genres"
    
    """
    url = f"https://shazam-core.p.rapidapi.com/v1/frame/cities"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam-core.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_track_details(track_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a details of track. Ex: https://www.shazam.com/track/216314/let-it-be"
    track_id: Track id
        
    """
    url = f"https://shazam-core.p.rapidapi.com/v2/tracks/details"
    querystring = {'track_id': track_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam-core.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def track_youtube_video(track_id: int, name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a youtube video for track. Search youtube video by name"
    track_id: Track id
        name: Video name
        
    """
    url = f"https://shazam-core.p.rapidapi.com/v1/tracks/youtube-video"
    querystring = {'track_id': track_id, 'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam-core.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def chart_by_city(city_id: int, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of tracks by city id. Ex: https://www.app.shazam.com/charts/top-50/austria/feldkirch"
    city_id: City id
        offset: Offset
        
    """
    url = f"https://shazam-core.p.rapidapi.com/v1/charts/city"
    querystring = {'city_id': city_id, }
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam-core.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_artist_details(artist_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a details of artist. Ex: https://www.shazam.com/artist/the-beatles/136975"
    artist_id: Artist id
        
    """
    url = f"https://shazam-core.p.rapidapi.com/v2/artists/details"
    querystring = {'artist_id': artist_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam-core.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tracks_related(track_id: int, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of tracks related by track id"
    track_id: Track id
        offset: Offset
        
    """
    url = f"https://shazam-core.p.rapidapi.com/v1/tracks/related"
    querystring = {'track_id': track_id, }
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam-core.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def world_chart(offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of tracks in word. Ex: https://www.app.shazam.com/charts/top-200/world"
    offset: Offset
        
    """
    url = f"https://shazam-core.p.rapidapi.com/v1/charts/world"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam-core.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_details(event_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a Events details"
    event_id: Event id
        
    """
    url = f"https://shazam-core.p.rapidapi.com/v1/events/details"
    querystring = {'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam-core.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_list(artist_id: int, date_from: str, page_number: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of Events by Artist ID"
    artist_id: Artist id
        date_from: Date from
        page_number: Page number
        
    """
    url = f"https://shazam-core.p.rapidapi.com/v1/events/list"
    querystring = {'artist_id': artist_id, 'date_from': date_from, }
    if page_number:
        querystring['page_number'] = page_number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam-core.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

