import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def shazam_events_list(artistid: str, l: str='en-US', is_from: str='2022-12-31', limit: int=50, offset: int=0, to: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List future events"
    artistid: The value of 'artist->adamid' field returned in …/search OR …/songs/v2/detect OR …/songs/get-details endpoint
The value of 'artists->id' field returned in …/shazam-songs/get-details OR …/albums/get-details OR …/albums/get-related-artist
        l: The language code
        is_from: The date to list events from
        limit: The number of items per page, for paging purpose
        offset: The page index, for paging purpose
        to: The date to list events to. Ex : 2023-01-15
        
    """
    url = f"https://shazam.p.rapidapi.com/shazam-events/list"
    querystring = {'artistId': artistid, }
    if l:
        querystring['l'] = l
    if is_from:
        querystring['from'] = is_from
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if to:
        querystring['to'] = to
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def artists_get_latest_release(is_id: str, l: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get latest release of an artist"
    id: The value of 'artist->adamid' field returned in .../search OR .../songs/v2/detect OR .../songs/get-details endpoint
The value of 'artists->id' field returned in .../shazam-songs/get-details OR .../albums/get-details OR .../albums/get-related-artist
        l: The language code
        
    """
    url = f"https://shazam.p.rapidapi.com/artists/get-latest-release"
    querystring = {'id': is_id, }
    if l:
        querystring['l'] = l
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def artists_get_summary(is_id: str, l: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get summary information related to an artist"
    id: The value of 'artist->adamid' field returned in .../search OR .../songs/v2/detect OR .../songs/get-details endpoint
The value of 'artists->id' field returned in .../shazam-songs/get-details OR .../albums/get-details OR .../albums/get-related-artist
        l: The language code
        
    """
    url = f"https://shazam.p.rapidapi.com/artists/get-summary"
    querystring = {'id': is_id, }
    if l:
        querystring['l'] = l
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def songs_get_related_artist(is_id: str, l: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get artist related to a song"
    id: The value of 'songs->id' field returned from .../shazam-songs/get-details endpoint
        l: The language code
        
    """
    url = f"https://shazam.p.rapidapi.com/songs/get-related-artist"
    querystring = {'id': is_id, }
    if l:
        querystring['l'] = l
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def songs_v2_get_details(is_id: str, l: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get details information of specific song"
    id: The value of 'songs->id' field returned from .../shazam-songs/get-details endpoint
        l: The language code
        
    """
    url = f"https://shazam.p.rapidapi.com/songs/v2/get-details"
    querystring = {'id': is_id, }
    if l:
        querystring['l'] = l
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def songs_get_details(key: str, locale: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get details information of specific song"
    key: The key field returned from .../songs/detect or .../search endpoint
        locale: The language code
        
    """
    url = f"https://shazam.p.rapidapi.com/songs/get-details"
    querystring = {'key': key, }
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def songs_get_count(key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get total times the specific song is detected/shazamed by using .../songs/detect or .../songs/v2/detect endpoint"
    key: The key field returned from .../songs/detect or .../search endpoint
        
    """
    url = f"https://shazam.p.rapidapi.com/songs/get-count"
    querystring = {'key': key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def songs_list_recommendations(key: str, locale: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List related ones to a specific song."
    key: The key field returned from .../songs/detect or .../search endpoint
        locale: The language code
        
    """
    url = f"https://shazam.p.rapidapi.com/songs/list-recommendations"
    querystring = {'key': key, }
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def songs_list_artist_top_tracks_deprecated(is_id: str, locale: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List top tracks of specific artist"
    id: The id field inside artists json object returned from .../songs/detect or .../search endpoint
        locale: The language code
        
    """
    url = f"https://shazam.p.rapidapi.com/songs/list-artist-top-tracks"
    querystring = {'id': is_id, }
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def albums_get_related_artist(is_id: str, l: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get artist related to an album"
    id: The value of 'albums->id' field returned in .../shazam-songs/get-details OR .../artists/get-albums OR .../artists/get-summary
The value of 'id' field returned in .../artists/get-albums OR .../artists/get-latest-release endpoint
The value of 'albumadamid' field returned in .../songs/v2/detect OR .../songs/detect endpoint
        l: The language code
        
    """
    url = f"https://shazam.p.rapidapi.com/albums/get-related-artist"
    querystring = {'id': is_id, }
    if l:
        querystring['l'] = l
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def albums_get_details(is_id: str, l: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed album of an album"
    id: The value of 'albums->id' field returned in .../shazam-songs/get-details OR .../artists/get-albums OR .../artists/get-summary
The value of 'id' field returned in .../artists/get-albums OR .../artists/get-latest-release endpoint
The value of 'albumadamid' field returned in .../songs/v2/detect OR .../songs/detect endpoint
        l: The language code
        
    """
    url = f"https://shazam.p.rapidapi.com/albums/get-details"
    querystring = {'id': is_id, }
    if l:
        querystring['l'] = l
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shazam_songs_list_similarities(is_id: str, locale: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get similar (You may also like) songs"
    id: The value of 'related-tracks->id' field returned in .../shazam-songs/get-details endpoint
        locale: The language code
        
    """
    url = f"https://shazam.p.rapidapi.com/shazam-songs/list-similarities"
    querystring = {'id': is_id, }
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shazam_songs_get_details(is_id: str, locale: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get mapping id information between systems to use with other endpoints."
    id: The value of 'id' field returned in .../search endpoint
        locale: The language code
        
    """
    url = f"https://shazam.p.rapidapi.com/shazam-songs/get-details"
    querystring = {'id': is_id, }
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def artists_get_top_songs(is_id: str, l: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get top songs of an artist"
    id: The value of 'artist->adamid' field returned in .../search OR .../songs/v2/detect OR .../songs/get-details endpoint
The value of 'artists->id' field returned in .../shazam-songs/get-details OR .../albums/get-details OR .../albums/get-related-artist
        l: The language code
        
    """
    url = f"https://shazam.p.rapidapi.com/artists/get-top-songs"
    querystring = {'id': is_id, }
    if l:
        querystring['l'] = l
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def artists_get_details(is_id: str, l: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed information of an artist"
    id: The value of 'artist->adamid' field returned in .../search OR .../songs/v2/detect OR .../songs/get-details endpoint
The value of 'artists->id' field returned in .../shazam-songs/get-details OR .../albums/get-details OR .../albums/get-related-artist
        l: The language code
        
    """
    url = f"https://shazam.p.rapidapi.com/artists/get-details"
    querystring = {'id': is_id, }
    if l:
        querystring['l'] = l
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def charts_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all available charts by cities, countries, and genres"
    
    """
    url = f"https://shazam.p.rapidapi.com/charts/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(term: str, locale: str='en-US', limit: int=5, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for songs, artists that match input term"
    term: Full name of songs or artists
        locale: The language code
        limit: For paging purpose, maximum is fixed at 5 items per response
        offset: For paging purpose
        
    """
    url = f"https://shazam.p.rapidapi.com/search"
    querystring = {'term': term, }
    if locale:
        querystring['locale'] = locale
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def auto_complete(term: str, locale: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get suggestions by word or phrase"
    term: Any word or phrase of song, artist, etc... that you are familiar with
        locale: The language code
        
    """
    url = f"https://shazam.p.rapidapi.com/auto-complete"
    querystring = {'term': term, }
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def charts_track(listid: str=None, pagesize: int=20, startfrom: int=0, locale: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all popular songs in specific chart"
    listid: The value of listId field returned in .../charts/list endpoint
        pagesize: Used for paging purpose, 20 items per response is maximum.
        startfrom: Used for paging purpose.
        locale: The language code
        
    """
    url = f"https://shazam.p.rapidapi.com/charts/track"
    querystring = {}
    if listid:
        querystring['listId'] = listid
    if pagesize:
        querystring['pageSize'] = pagesize
    if startfrom:
        querystring['startFrom'] = startfrom
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shazam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

