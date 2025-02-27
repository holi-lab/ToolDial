import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_watch_playlist(video_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a watch list of tracks. This watch playlist appears when you press play on a track in YouTube Music.
		
		
		List of watch playlist items. The counterpart key is optional and only appears if a song has a corresponding video counterpart (UI song/video switcher)."
    
    """
    url = f"https://youtube-music-api-detailed.p.rapidapi.com/get_watch_playlist"
    querystring = {'video_id': video_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-music-api-detailed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_lyrics(browse_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns lyrics of a song or video."
    
    """
    url = f"https://youtube-music-api-detailed.p.rapidapi.com/get_lyrics"
    querystring = {'browse_id': browse_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-music-api-detailed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_song_related(browse_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets related content for a song. Equivalent to the content shown in the “Related” tab of the watch panel."
    
    """
    url = f"https://youtube-music-api-detailed.p.rapidapi.com/get_song_related"
    querystring = {'browse_id': browse_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-music-api-detailed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_song(video_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns metadata and streaming information about a song or video."
    
    """
    url = f"https://youtube-music-api-detailed.p.rapidapi.com/get_song"
    querystring = {'video_id': video_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-music-api-detailed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_playlists(params: str, user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a list of playlists for a given user. Call this function again with the returned params to get the full list."
    
    """
    url = f"https://youtube-music-api-detailed.p.rapidapi.com/get_user_playlists"
    querystring = {'params': params, 'user_id': user_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-music-api-detailed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user(user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a user’s page. A user may own videos or playlists."
    user_id: channelId of the user
        
    """
    url = f"https://youtube-music-api-detailed.p.rapidapi.com/get_user"
    querystring = {'user_id': user_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-music-api-detailed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_album(browse_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information and tracks of an album."
    browse_id: browseId – browseId of the album, for example returned by **search()**
        
    """
    url = f"https://youtube-music-api-detailed.p.rapidapi.com/get_album"
    querystring = {'browse_id': browse_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-music-api-detailed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_artist_albums(channel_id: str, params: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about an artist and their top releases (songs, albums, singles, videos, and related artists). The top lists contain pointers for getting the full list of releases"
    params: \\\"params\\\" data obtained by **get_artist()**
        
    """
    url = f"https://youtube-music-api-detailed.p.rapidapi.com/get_artist_albums"
    querystring = {'channel_id': channel_id, 'params': params, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-music-api-detailed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_artist(channel_id: str='UCedvOgsKFzcK3hA5taf3KoQ', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about an artist and their top releases (songs, albums, singles, videos, and related artists). The top lists contain pointers for getting the full list of releases."
    
    """
    url = f"https://youtube-music-api-detailed.p.rapidapi.com/get_artist"
    querystring = {}
    if channel_id:
        querystring['channel_id'] = channel_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-music-api-detailed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trends(country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get latest charts data from YouTube Music: Top songs, top videos, top artists and top trending videos. Global charts have no Trending section, US charts have an extra Genres section with some Genre charts."
    country: ISO 3166-1 Alpha-2 country code. Default: ZZ = Global
https://www.iban.com/country-codes
        
    """
    url = f"https://youtube-music-api-detailed.p.rapidapi.com/popular/trends"
    querystring = {'country': country, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-music-api-detailed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mood_categories(lang: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns mood categories."
    
    """
    url = f"https://youtube-music-api-detailed.p.rapidapi.com/popular/categories"
    querystring = {'lang': lang, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-music-api-detailed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(lang: str, query: str, limit: int=40, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search YouTube music Returns results within the provided category."
    
    """
    url = f"https://youtube-music-api-detailed.p.rapidapi.com/popular/search"
    querystring = {'lang': lang, 'query': query, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-music-api-detailed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

