import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def download_lyrics_on_spotify_2_quotas(track: str, format: str=None, candidate: int=None, delay: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides a solution to download a track's lyrics on Spotify. (Powered by [Spotify Scraper](https://rapidapi.com/DataFanatic/api/spotify-scraper/))"
    track: Track URL or ID.
        format: File format. Defaults to `lrc`.
        candidate: The number of the most related Spotify tracks among which a track with the nearest duration will be selected to download its lyrics. Defaults to `3`.
        delay: Delay time in milliseconds. Defaults to `0`. Negative value is allowed.
        
    """
    url = f"https://soundcloud-scraper.p.rapidapi.com/v1/track/lyrics"
    querystring = {'track': track, }
    if format:
        querystring['format'] = format
    if candidate:
        querystring['candidate'] = candidate
    if delay:
        querystring['delay'] = delay
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soundcloud-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download_lyrics_on_spotify_2_quotas(track: str, format: str=None, candidate: int=None, delay: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides a solution to download a track's lyrics on Spotify. (Powered by [Spotify Scraper](https://rapidapi.com/DataFanatic/api/spotify-scraper/))"
    track: Track URL or ID.
        format: File format. Defaults to `lrc`.
        candidate: The number of the most related Spotify tracks among which a track with the nearest duration will be selected to download its lyrics. Defaults to `3`.
        delay: Delay time in milliseconds. Defaults to `0`. Negative value is allowed.
        
    """
    url = f"https://soundcloud-scraper.p.rapidapi.com/v1/track/lyrics"
    querystring = {'track': track, }
    if format:
        querystring['format'] = format
    if candidate:
        querystring['candidate'] = candidate
    if delay:
        querystring['delay'] = delay
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soundcloud-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_track_metadata_1_3_quotas(track: str, download: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint fetches full details of a track on SoundCloud and helps you download uncut tracks of high and standard quality. (Tutorial: [How to Use Audio URLs](https://rapidapi.com/DataFanatic/api/soundcloud-scraper/tutorials/how-to-use-audio-urls))"
    track: Track URL or ID.
        download: Download quality. Defaults to `sq`.
* **`hq`**: High and standard quality - 3 quotas
* **`sq`**: Standard quality - 2 quotas
* **`none`**: No download - 1 quota

The final price depends on the existence of the file of the corresponding quality. For example, if `hq` is requested but the high-quality file does not exist, then the price will be 2 quotas if the standard one exists or 1 quota if there is no file available.
        
    """
    url = f"https://soundcloud-scraper.p.rapidapi.com/v1/track/metadata"
    querystring = {'track': track, }
    if download:
        querystring['download'] = download
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soundcloud-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_for_albums(term: str, offset: int=None, genreortag: str=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint searches for SoundCloud albums with optional filters. Pagination scraping is supported."
    term: Search term.
        offset: (**README**: [How to Use Endpoints with `offset`](https://rapidapi.com/DataFanatic/api/soundcloud-scraper/tutorials/how-to-use-endpoints-with-%60offset%60))
The number of items omitted before the results. Defaults to `0`.
        genreortag: Filters by genre or tag (e.g., `classical`).
        limit: The max number of items returned. Defaults to `50`.
        
    """
    url = f"https://soundcloud-scraper.p.rapidapi.com/v1/search/albums"
    querystring = {'term': term, }
    if offset:
        querystring['offset'] = offset
    if genreortag:
        querystring['genreOrTag'] = genreortag
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soundcloud-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_for_users(term: str, offset: int=None, limit: int=None, place: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint searches for SoundCloud users with optional filters. Pagination scraping is supported."
    term: Search term.
        offset: (**README**: [How to Use Endpoints with `offset`](https://rapidapi.com/DataFanatic/api/soundcloud-scraper/tutorials/how-to-use-endpoints-with-%60offset%60))
The number of items omitted before the results. Defaults to `0`.
        limit: The max number of items returned. Defaults to `50`.
        place: Filters by a user's location (e.g., `London`).
        
    """
    url = f"https://soundcloud-scraper.p.rapidapi.com/v1/search/users"
    querystring = {'term': term, }
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if place:
        querystring['place'] = place
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soundcloud-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_for_playlists(term: str, offset: int=None, limit: int=None, genreortag: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint searches for SoundCloud playlists with optional filters. Pagination scraping is supported."
    term: Search term.
        offset: (**README**: [How to Use Endpoints with `offset`](https://rapidapi.com/DataFanatic/api/soundcloud-scraper/tutorials/how-to-use-endpoints-with-%60offset%60))
The number of items omitted before the results. Defaults to `0`.
        limit: The max number of items returned. Defaults to `50`.
        genreortag: Filters by genre or tag (e.g., `classical`).
        
    """
    url = f"https://soundcloud-scraper.p.rapidapi.com/v1/search/playlists"
    querystring = {'term': term, }
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if genreortag:
        querystring['genreOrTag'] = genreortag
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soundcloud-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_user_followers(limit: int=None, offsettoken: str=None, user: str='https://soundcloud.com/atlantic-records-uk', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists a user's followers."
    limit: The max number of items returned. Defaults to `50`.
        offsettoken: (**README**: [How to Use Endpoints with `offsetToken`](https://rapidapi.com/DataFanatic/api/soundcloud-scraper/tutorials/how-to-use-endpoints-with-%60offsettoken%60))
A string for getting the next page of data. If not specified, the first page of data will be returned. If specified, `user` will be ignored.
        user: User URL or ID.
        
    """
    url = f"https://soundcloud-scraper.p.rapidapi.com/v1/user/followers"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offsettoken:
        querystring['offsetToken'] = offsettoken
    if user:
        querystring['user'] = user
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soundcloud-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_user_reposts_tracks_playlists(user: str='https://soundcloud.com/atlantic-records-uk', limit: int=None, offsettoken: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists reposts of a user. Pagination scraping is supported."
    user: User URL or ID.
        limit: The max number of items returned. Defaults to `50`.
        offsettoken: (**README**: [How to Use Endpoints with `offsetToken`](https://rapidapi.com/DataFanatic/api/soundcloud-scraper/tutorials/how-to-use-endpoints-with-%60offsettoken%60))
A string for getting the next page of data. If not specified, the first page of data will be returned. If specified, `user` will be ignored.
        
    """
    url = f"https://soundcloud-scraper.p.rapidapi.com/v1/user/reposts"
    querystring = {}
    if user:
        querystring['user'] = user
    if limit:
        querystring['limit'] = limit
    if offsettoken:
        querystring['offsetToken'] = offsettoken
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soundcloud-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_user_playlists(user: str, offset: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists the playlists of a user. Pagination scraping is supported."
    user: User URL or ID.
        offset: (**README**: [How to Use Endpoints with `offset`](https://rapidapi.com/DataFanatic/api/soundcloud-scraper/tutorials/how-to-use-endpoints-with-%60offset%60))
The number of items omitted before the results. Defaults to `0`.
        limit: The max number of items returned. Defaults to `50`.
        
    """
    url = f"https://soundcloud-scraper.p.rapidapi.com/v1/user/playlists"
    querystring = {'user': user, }
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soundcloud-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_user_popular_tracks(user: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists all popular tracks of a user."
    user: User URL or ID.
        
    """
    url = f"https://soundcloud-scraper.p.rapidapi.com/v1/user/toptracks"
    querystring = {'user': user, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soundcloud-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_user_likes_tracks_playlists(offsettoken: str=None, user: str='https://soundcloud.com/atlantic-records-uk', limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists tracks, playlists and albums liked by the specified user."
    offsettoken: (**README**: [How to Use Endpoints with `offsetToken`](https://rapidapi.com/DataFanatic/api/soundcloud-scraper/tutorials/how-to-use-endpoints-with-%60offsettoken%60))
A string for getting the next page of data. If not specified, the first page of data will be returned. If specified, `user` will be ignored.
        user: User URL or ID.
        limit: The max number of items returned. Defaults to `50`.
        
    """
    url = f"https://soundcloud-scraper.p.rapidapi.com/v1/user/likes"
    querystring = {}
    if offsettoken:
        querystring['offsetToken'] = offsettoken
    if user:
        querystring['user'] = user
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soundcloud-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_user_comments(offsettoken: str=None, user: str='https://soundcloud.com/user-999824827', limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists a user's comments."
    offsettoken: (**README**: [How to Use Endpoints with `offsetToken`](https://rapidapi.com/DataFanatic/api/soundcloud-scraper/tutorials/how-to-use-endpoints-with-%60offsettoken%60))
A string for getting the next page of data. If not specified, the first page of data will be returned. If specified, `user` will be ignored.
        user: User URL or ID.
        limit: The max number of items returned. Defaults to `50`.
        
    """
    url = f"https://soundcloud-scraper.p.rapidapi.com/v1/user/comments"
    querystring = {}
    if offsettoken:
        querystring['offsetToken'] = offsettoken
    if user:
        querystring['user'] = user
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soundcloud-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_track_comments(track: str, filterreplies: bool=None, offset: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists comments of a track. Pagination scraping is supported."
    track: Track URL or ID.
        filterreplies: Whether to remove replies. Defaults to `false`.
        offset: (**README**: [How to Use Endpoints with `offset`](https://rapidapi.com/DataFanatic/api/soundcloud-scraper/tutorials/how-to-use-endpoints-with-%60offset%60))
The number of items omitted before the results. Defaults to `0`.
        limit: The max number of items returned. Defaults to `50`.
        
    """
    url = f"https://soundcloud-scraper.p.rapidapi.com/v1/track/comments"
    querystring = {'track': track, }
    if filterreplies:
        querystring['filterReplies'] = filterreplies
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soundcloud-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_for_go_tracks(term: str, offset: int=None, license: str=None, goplus: bool=None, limit: int=None, duration: str=None, addedin: str=None, genreortag: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint searches for SoundCloud (Go+) tracks with optional filters. Pagination scraping is supported."
    term: Search term.
        offset: (**README**: [How to Use Endpoints with `offset`](https://rapidapi.com/DataFanatic/api/soundcloud-scraper/tutorials/how-to-use-endpoints-with-%60offset%60))
The number of items omitted before the results. Defaults to `0`.
        license: Filters by how a track can be used. Defaults to `listen`.
* **`listen`**: To listen to
* **`share`**: To share
* **`use`**: To use commercially
* **`modify`**: To modify commercially
        goplus: Searches for Go+ tracks only. Defaults to `false`.
        limit: The max number of items returned. Defaults to `50`.
        duration: Filters by a track's duration. Defaults to `any`.
* **`any`**: Any duration
* **`short`**: < 2 min
* **`medium`**: 2-10 min
* **`long`**: 10-30 min
* **`epic`**: > 30 min
        addedin: Filters by when a track was added. Defaults to `any`.
* **`any`**: Added at any time
* **`hour`**: Added in the past hour
* **`day`**: Added in the past day
* **`week`**: Added in the past week
* **`month`**: Added in the past month
* **`year`**: Added in the past year
        genreortag: Filters by genre or tag (e.g., `classical`).
        
    """
    url = f"https://soundcloud-scraper.p.rapidapi.com/v1/search/tracks"
    querystring = {'term': term, }
    if offset:
        querystring['offset'] = offset
    if license:
        querystring['license'] = license
    if goplus:
        querystring['goPlus'] = goplus
    if limit:
        querystring['limit'] = limit
    if duration:
        querystring['duration'] = duration
    if addedin:
        querystring['addedIn'] = addedin
    if genreortag:
        querystring['genreOrTag'] = genreortag
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soundcloud-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_user_followings(limit: int=None, user: str='https://soundcloud.com/lovely-days-press', offsettoken: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists users that follow the specified user."
    limit: The max number of items returned. Defaults to `50`.
        user: User URL or ID.
        offsettoken: (**README**: [How to Use Endpoints with `offsetToken`](https://rapidapi.com/DataFanatic/api/soundcloud-scraper/tutorials/how-to-use-endpoints-with-%60offsettoken%60))
A string for getting the next page of data. If not specified, the first page of data will be returned. If specified, `user` will be ignored.
        
    """
    url = f"https://soundcloud-scraper.p.rapidapi.com/v1/user/followings"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if user:
        querystring['user'] = user
    if offsettoken:
        querystring['offsetToken'] = offsettoken
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soundcloud-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_user_albums(user: str, limit: int=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists the albums of a user. Pagination scraping is supported."
    user: User URL or ID.
        limit: The max number of items returned. Defaults to `50`.
        offset: (**README**: [How to Use Endpoints with `offset`](https://rapidapi.com/DataFanatic/api/soundcloud-scraper/tutorials/how-to-use-endpoints-with-%60offset%60))
The number of items omitted before the results. Defaults to `0`.
        
    """
    url = f"https://soundcloud-scraper.p.rapidapi.com/v1/user/albums"
    querystring = {'user': user, }
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soundcloud-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_user_tracks(user: str='https://soundcloud.com/edsheeran', limit: int=None, offsettoken: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists tracks of a user. Pagination scraping is supported."
    user: User URL or ID.
        limit: The max number of items returned. Defaults to `50`.
        offsettoken: (**README**: [How to Use Endpoints with `offsetToken`](https://rapidapi.com/DataFanatic/api/soundcloud-scraper/tutorials/how-to-use-endpoints-with-%60offsettoken%60))
A string for getting the next page of data. If not specified, the first page of data will be returned. If specified, `user` will be ignored.
        
    """
    url = f"https://soundcloud-scraper.p.rapidapi.com/v1/user/tracks"
    querystring = {}
    if user:
        querystring['user'] = user
    if limit:
        querystring['limit'] = limit
    if offsettoken:
        querystring['offsetToken'] = offsettoken
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soundcloud-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_profile(user: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint fetches a user's profile."
    user: User URL or ID.
        
    """
    url = f"https://soundcloud-scraper.p.rapidapi.com/v1/user/profile"
    querystring = {'user': user, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soundcloud-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_station_details(stationurl: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint fetches full details of a SoundCloud station, including all tracks."
    stationurl: Station URL.
        
    """
    url = f"https://soundcloud-scraper.p.rapidapi.com/v1/station/details"
    querystring = {'stationUrl': stationurl, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soundcloud-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_playlist_album_tracks(playlist: str, limit: int=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists tracks of a playlist or album. Pagination scraping is supported."
    playlist: Playlist/album URL or ID.
        limit: The max number of items returned. Defaults to `50`.
        offset: (**README**: [How to Use Endpoints with `offset`](https://rapidapi.com/DataFanatic/api/soundcloud-scraper/tutorials/how-to-use-endpoints-with-%60offset%60))
The number of items omitted before the results. Defaults to `0`.
        
    """
    url = f"https://soundcloud-scraper.p.rapidapi.com/v1/playlist/tracks"
    querystring = {'playlist': playlist, }
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soundcloud-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_playlist_album_metadata(playlist: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint fetches full details of a playlist or album, including all track IDs."
    playlist: Playlist/album URL or ID.
        
    """
    url = f"https://soundcloud-scraper.p.rapidapi.com/v1/playlist/metadata"
    querystring = {'playlist': playlist, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soundcloud-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_home_page_overview(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint fetches trending information displayed on the SoundCloud home page as an incognito visitor."
    
    """
    url = f"https://soundcloud-scraper.p.rapidapi.com/v1/home/details"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soundcloud-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

