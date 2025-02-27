import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def popular_artists(limit: int=20, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of artists popular on Humm; returns a list of artist objects"
    limit: Number of returned results (20 by default)
        offset: Offset results by said number (0 by default)
        
    """
    url = f"https://humm-api.p.rapidapi.com/artists/popular"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "humm-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_artists(keyword: str, limit: int=20, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for an artist; returns a list of artist objects"
    keyword: Search text
        limit: Number of returned results (20 by default)
        offset: Offset results by said number (0 by default)
        
    """
    url = f"https://humm-api.p.rapidapi.com/artists"
    querystring = {'keyword': keyword, }
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "humm-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def recent_artists(offset: int=0, limit: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of artists recently added on Humm; returns a list of artist objects"
    offset: Offset results by said number (0 by default)
        limit: Number of returned results (20 by default)
        
    """
    url = f"https://humm-api.p.rapidapi.com/artists/recent"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "humm-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def featured_artists(offset: int=0, genre: str=None, limit: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of artists featured by Humm; returns a list of artist objects"
    offset: Offset results by said number (0 by default)
        genre: Filter results by genre
        limit: Number of returned results (20 by default)
        
    """
    url = f"https://humm-api.p.rapidapi.com/artists/featured"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if genre:
        querystring['genre'] = genre
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "humm-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def artist_top_songs(is_id: str, limit: int=20, offset: int=0, songtype: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of an artist's top songs; returns a list of song objects for a given artist id"
    id: Unique identifier of artist (ex. Blur)
        limit: Number of returned results (20 by default)
        offset: Offset results by said number (0 by default)
        songtype: Filter by type of song: track (default), cover, video.
        
    """
    url = f"https://humm-api.p.rapidapi.com/artists/{is_id}/topsongs"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if songtype:
        querystring['songtype'] = songtype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "humm-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def artist_playlists_albums(is_id: str, limit: int=20, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get playlists / albums associated with an artist; returns a list of playlist / album objects for a given id"
    id: Unique identifier of artist (ex. Blur)
        limit: Number of returned results (20 by default)
        offset: Offset results by said number (0 by default)
        
    """
    url = f"https://humm-api.p.rapidapi.com/artists/{is_id}/playlists"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "humm-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def artist_details(is_id: str, limit: int=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get an artist; returns artist object for a given id"
    id: Unique identifier of artist (ex. Blur)
        offset: Offset results by said number (0 by default)
        
    """
    url = f"https://humm-api.p.rapidapi.com/artists/{is_id}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "humm-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_details(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a user; returns a user object for a given id"
    id: Unique identifier of user (ex. user1)
        
    """
    url = f"https://humm-api.p.rapidapi.com/users/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "humm-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def artist_similar(is_id: str, limit: int=None, offset: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of musically similar artists; returns a list of artist objects for a given id"
    id: Unique identifier of artist (ex. Blur)
        limit: Number of returned results (20 by default)
        offset: Offset results by said number (0 by default)
        
    """
    url = f"https://humm-api.p.rapidapi.com/artists/{is_id}/similar"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "humm-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_playlists(is_id: str, limit: int=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of user owned playlists; returns list of playlist objects for given user id"
    id: Unique identifier of user (ex. user1)
        limit: Number of returned results (20 by default)
        offset: Offset results by said number (0 by default)
        
    """
    url = f"https://humm-api.p.rapidapi.com/users/{is_id}/playlists"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "humm-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def discover_artists(limit: int=20, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of recommended artists; returns a list of artist objects"
    limit: Number of returned results (20 by default)
        offset: Offset results by said number (0 by default)
        
    """
    url = f"https://humm-api.p.rapidapi.com/users/me/discover/artists"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "humm-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def discover_album_releases(limit: int=20, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of recommended albums; returns a list of album objects"
    limit: Number of returned results (20 by default)
        offset: Offset results by said number (0 by default)
        
    """
    url = f"https://humm-api.p.rapidapi.com/users/me/discover/releases"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "humm-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_songs(keyword: str, limit: int=20, offset: int=0, songtype: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for a song; returns a list of song objects"
    keyword: Search text (title or artist)
        limit: Number of returned results (20 by default)
        offset: Offset results by said number (0 by default)
        songtype: Filter by type of song: track (default), cover, video.
        
    """
    url = f"https://humm-api.p.rapidapi.com/songs"
    querystring = {'keyword': keyword, }
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if songtype:
        querystring['songtype'] = songtype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "humm-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def popular_songs(limit: int=20, offset: int=0, genre: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of songs popular on Humm; returns a list of song objects"
    limit: Number of returned results (20 by default)
        offset: Offset results by said number (0 by default)
        genre: Filter by genre: rock, pop, indie pop, electronic, hip hop, hip house, metal, blues, classical, bossa nova, cantautor, hard rock, punk, soundtrack, new age, honky tonk, flamenco, samba, salsa, ska, world, poetry, beats & rhymes, motivation, soul, folk, reggae, latin, jazz, dance, ambient, romantic, workout, easy listening, chill-out, a cappella
        
    """
    url = f"https://humm-api.p.rapidapi.com/songs/popular"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if genre:
        querystring['genre'] = genre
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "humm-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def featured_songs(limit: int=20, offset: int=0, genre: str='blues', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a llist of songs featured by Humm; returns a list of song objects"
    limit: Number of returned results (20 by default)
        offset: Offset results by said number (0 by default)
        genre: Filter by genre: rock, pop, indie pop, electronic, hip hop, hip house, metal, blues, classical, bossa nova, cantautor, hard rock, punk, soundtrack, new age, honky tonk, flamenco, samba, salsa, ska, world, poetry, beats & rhymes, motivation, soul, folk, reggae, latin, jazz, dance, ambient, romantic, workout, easy listening, chill-out, a cappella
        
    """
    url = f"https://humm-api.p.rapidapi.com/songs/featured"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if genre:
        querystring['genre'] = genre
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "humm-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def song_similar(is_id: str, limit: int=20, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of similar songs; returns a list of song objects for a given id"
    id: Unique identifier of song (ex. Airbag by Radiohead)
        limit: Number of returned results (20 by default)
        offset: Offset results by said number (0 by default)
        
    """
    url = f"https://humm-api.p.rapidapi.com/songs/{is_id}/similar"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "humm-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def song_details(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a song; returns song object for given id"
    id: Unique identifier of song (ex. Airbag by Radiohead)
        
    """
    url = f"https://humm-api.p.rapidapi.com/songs/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "humm-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def recent_songs(limit: int=20, offset: int=0, genre: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of songs recently added on Humm; returns a list of song objects"
    limit: Number of returned results (20 by default)
        offset: Offset results by said number (0 by default)
        genre: Filter by genre: rock, pop, indie pop, electronic, hip hop, hip house, metal, blues, classical, bossa nova, cantautor, hard rock, punk, soundtrack, new age, honky tonk, flamenco, samba, salsa, ska, world, poetry, beats & rhymes, motivation, soul, folk, reggae, latin, jazz, dance, ambient, romantic, workout, easy listening, chill-out, a cappella
        
    """
    url = f"https://humm-api.p.rapidapi.com/songs/recent"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if genre:
        querystring['genre'] = genre
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "humm-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def song_appears_in(is_id: str, limit: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of playlists a song appears in; returns a list of playlist objects for a given id"
    id: Unique identifier of song (ex. Song 2 by Blur)
        limit: Number of returned results (20 by default)
        
    """
    url = f"https://humm-api.p.rapidapi.com/songs/{is_id}/appearsin"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "humm-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def discover_playlists_albums(limit: int=20, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of recommended playlists / albums; returns a list of playlist / album objects"
    limit: Number of returned results (20 by default)
        offset: Offset results by said number (0 by default)
        
    """
    url = f"https://humm-api.p.rapidapi.com/users/me/discover/playlists"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "humm-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def popular_playlists(limit: int=20, offset: int=0, selection: str=None, uid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of playlists popular on Humm; returns a list of playlist objects (for an optional selection or user id)"
    limit: Number of returned results (20 by default)
        offset: Offset results by said number (0 by default)
        selection: Filter by "featured" (popular featured playlists), "own" (popular playlists for given user), "following" (popular playlists within given user subscriptions)
        uid: Unique identifier of user
        
    """
    url = f"https://humm-api.p.rapidapi.com/playlists/popular"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if selection:
        querystring['selection'] = selection
    if uid:
        querystring['uid'] = uid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "humm-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_playlists_albums(keyword: str, limit: int=20, offset: int=0, album: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Playlists / Albums; returns a list of playlist / album objects"
    keyword: Search text
        limit: Number of returned results (20 by default)
        offset: Offset results by said number (0 by default)
        album: Search an album (true) or a playlist (false; default)
        
    """
    url = f"https://humm-api.p.rapidapi.com/playlists"
    querystring = {'keyword': keyword, }
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if album:
        querystring['album'] = album
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "humm-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def featured_playlists(offset: int=0, limit: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of playlists featured by Humm; returns a list of playlist objects"
    offset: Offset results by said number (0 by default)
        limit: Number of returned results (20 by default)
        
    """
    url = f"https://humm-api.p.rapidapi.com/playlists/featured"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "humm-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def recent_playlists(limit: int=20, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of playlists recently added on Humm; returns a list of playlist objects"
    limit: Number of returned results (20 by default)
        offset: Offset results by said number (0 by default)
        
    """
    url = f"https://humm-api.p.rapidapi.com/playlists/recent"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "humm-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playlist_songs(is_id: str, limit: int=20, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of playlist songs; return a list of song objects for a given id"
    id: Unique identifier of playlist
        limit: Number of returned results (20 by default)
        offset: Offset results by said number (0 by default)
        
    """
    url = f"https://humm-api.p.rapidapi.com/playlists/{is_id}/songs"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "humm-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playlist_details(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a playlist; returns playlist object for a given id"
    id: Unique identifier of playlist
        
    """
    url = f"https://humm-api.p.rapidapi.com/playlists/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "humm-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def artist_radio(is_id: str, limit: int=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a curated list of artist songs; returns a list of song objects for a given artist id"
    id: Unique identifier of artist (ex. Blur)
        limit: Number of returned results (20 by default)
        offset: Offset results by said number (0 by default)
        
    """
    url = f"https://humm-api.p.rapidapi.com/artists/{is_id}/radio"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "humm-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def current_user(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the current user; returns a user object"
    
    """
    url = f"https://humm-api.p.rapidapi.com/users/me"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "humm-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_follows(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of the users a user is following; returns a list of user objects for a given id"
    id: Unique identifier of user (ex. user1)
        
    """
    url = f"https://humm-api.p.rapidapi.com/users/{is_id}/follows"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "humm-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def radio(limit: str=None, genres: str=None, moods: str=None, own: bool=None, discovery: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of songs for a radio; returns a list of song objects (*Must pass in one optional parameter e.g. 'rock')"
    limit: Number of returned results (20 by default)
        genres: String or list of strings
        moods: String or list of strings
        own: Current user's radio (false by default)
        discovery: Current user's discovery radio (false by default)
        
    """
    url = f"https://humm-api.p.rapidapi.com/radio"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if genres:
        querystring['genres'] = genres
    if moods:
        querystring['moods'] = moods
    if own:
        querystring['own'] = own
    if discovery:
        querystring['discovery'] = discovery
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "humm-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

