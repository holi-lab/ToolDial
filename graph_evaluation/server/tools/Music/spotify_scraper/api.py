import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def download_track_lyrics(trackid: str, delay: int=None, format: str=None, removenote: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows downloading lyrics in different formats. You can also tweak the delay time of lyrics."
    delay: Delay time in milliseconds. Defaults to `0`. Negative value is allowed.
        format: File format. Defaults to `lrc`.
        removenote: Whether to remove lines that only contain a single Eighth Note Emoji (♪). Defaults to `true`.
        
    """
    url = f"https://spotify-scraper.p.rapidapi.com/v1/track/lyrics"
    querystring = {'trackId': trackid, }
    if delay:
        querystring['delay'] = delay
    if format:
        querystring['format'] = format
    if removenote:
        querystring['removeNote'] = removenote
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download_track_lyrics(trackid: str, delay: int=None, format: str=None, removenote: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows downloading lyrics in different formats. You can also tweak the delay time of lyrics."
    delay: Delay time in milliseconds. Defaults to `0`. Negative value is allowed.
        format: File format. Defaults to `lrc`.
        removenote: Whether to remove lines that only contain a single Eighth Note Emoji (♪). Defaults to `true`.
        
    """
    url = f"https://spotify-scraper.p.rapidapi.com/v1/track/lyrics"
    querystring = {'trackId': trackid, }
    if delay:
        querystring['delay'] = delay
    if format:
        querystring['format'] = format
    if removenote:
        querystring['removeNote'] = removenote
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_track_metadata(trackid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint fetches full details of a track."
    trackid: Track ID.
        
    """
    url = f"https://spotify-scraper.p.rapidapi.com/v1/track/metadata"
    querystring = {'trackId': trackid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def resolve_short_url(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint resolves a Spotify short URL and returns a corresponding original URL."
    url: Spotify short URL.
        
    """
    url = f"https://spotify-scraper.p.rapidapi.com/v1/misc/resolve-short-url"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_playlist_metadata(playlistid: str, limit: int=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint fetches full details of a playlist."
    playlistid: Playlist ID.
        limit: The max number of items returned per request. Defaults to `0`.
        offset: The number of items omitted before the results. Defaults to `0`.
        
    """
    url = f"https://spotify-scraper.p.rapidapi.com/v1/playlist/metadata"
    querystring = {'playlistId': playlistid, }
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(term: str, offset: int=None, type: str=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint searches for albums, artists, episodes, genres, playlists, podcasts, and users. Pagination scraping is supported."
    term: Search Term.
        offset: The number of items omitted before the results. Defaults to `0`.
        type: Search type. Defaults to `all`.
        limit: The max number of items returned per request. Defaults to `50`.
        
    """
    url = f"https://spotify-scraper.p.rapidapi.com/v1/search"
    querystring = {'term': term, }
    if offset:
        querystring['offset'] = offset
    if type:
        querystring['type'] = type
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_artist_overview(artistid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint fetches full details of an artist."
    artistid: Artist ID.
        
    """
    url = f"https://spotify-scraper.p.rapidapi.com/v1/artist/overview"
    querystring = {'artistId': artistid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_track_id_by_name(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is a simplified version of the search endpoint that gets the ID of a track by its name."
    name: Track name. It's recommended to append the artist's name to avoid unexpected results.
        
    """
    url = f"https://spotify-scraper.p.rapidapi.com/v1/track/search"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download_track_on_youtube_3_quotas(track: str, candidate: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides a solution to download a Spotify track on YouTube and returns URLs of audio files. (Powered by [Youtube Media Downloader](https://rapidapi.com/DataFanatic/api/youtube-media-downloader/). Get a better price by subscribing to and using it directly.)"
    track: Track ID, track share URL, track name, or other search term.
        candidate: The number of the most related YouTube videos among which a video with the nearest duration will be selected to extract downloadable audio URLs. Defaults to `3`.
        
    """
    url = f"https://spotify-scraper.p.rapidapi.com/v1/track/download"
    querystring = {'track': track, }
    if candidate:
        querystring['candidate'] = candidate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download_track_on_soundcloud_3_4_quotas(track: str, quality: str=None, candidate: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides a solution to download a Spotify track on SoundCloud. HQ (4 Quotas) and SQ (3 Quotas) audios. (Tutorial: [How to Use SoundCloud Audio URLs](https://rapidapi.com/DataFanatic/api/soundcloud-scraper/tutorials/how-to-use-audio-urls)) (Powered by [SoundCloud Scraper](https://rapidapi.com/DataFanatic/api/soundcloud-scraper/). Get a better price by subscribing to and using it directly.)"
    track: Track ID, track share URL, track name, or other search term.
        quality: Download quality. Defaults to `sq`.
* **`hq`**: High and standard quality - 4 quotas
* **`sq`**: Standard quality - 3 quotas

The final price depends on the existence of the file of the corresponding quality. For example, if `hq` is requested but the high-quality file does not exist, then the price will be 3 quotas if the standard one exists.
        candidate: The number of the most related SoundCloud tracks among which a track with the nearest duration will be selected to generate downloadable audio URLs. Defaults to `3`.
        
    """
    url = f"https://spotify-scraper.p.rapidapi.com/v1/track/download/soundcloud"
    querystring = {'track': track, }
    if quality:
        querystring['quality'] = quality
    if candidate:
        querystring['candidate'] = candidate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_user_public_playlists(userid: str, limit: int=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists public playlists of a user. Pagination scraping is supported."
    userid: User ID.
        limit: The max number of items returned per request. Defaults to `200`.
        offset: The number of items omitted before the results. Defaults to `0`.
        
    """
    url = f"https://spotify-scraper.p.rapidapi.com/v1/user/playlists"
    querystring = {'userId': userid, }
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_profile(userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint fetches a user's profile."
    userid: User ID.
        
    """
    url = f"https://spotify-scraper.p.rapidapi.com/v1/user/profile"
    querystring = {'userId': userid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_user_recently_played_artists(userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists recently played artists of a user."
    userid: User ID.
        
    """
    url = f"https://spotify-scraper.p.rapidapi.com/v1/user/artists"
    querystring = {'userId': userid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_related_artists(artistid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists related artists of an artist."
    artistid: Artist ID.
        
    """
    url = f"https://spotify-scraper.p.rapidapi.com/v1/artist/related"
    querystring = {'artistId': artistid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_artist_playlists_featuring_discoveredon(artistid: str, type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists playlists associated with an artist."
    artistid: Artist ID.
        type: Playlist type. Defaults to `playlist`.
        
    """
    url = f"https://spotify-scraper.p.rapidapi.com/v1/artist/playlists"
    querystring = {'artistId': artistid, }
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_artist_albums_singles_compilations_appearson(artistid: str, offset: int=None, type: str=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists albums associated with an artist. Pagination scraping is supported."
    offset: The number of items omitted before the results. Defaults to `0`. This argument will be ignored if `type` is `appearsOn`.
        type: Album type. Defaults to `album`. Note that type `appearsOn` doesn't support pagination scraping.
        limit: The max number of items returned per request. Defaults to `50`. This argument will be ignored if `type` is `appearsOn`.
        
    """
    url = f"https://spotify-scraper.p.rapidapi.com/v1/artist/albums"
    querystring = {'artistId': artistid, }
    if offset:
        querystring['offset'] = offset
    if type:
        querystring['type'] = type
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_artist_concerts(artistid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists all concerts of an artist."
    artistid: Artist ID.
        
    """
    url = f"https://spotify-scraper.p.rapidapi.com/v1/artist/concerts"
    querystring = {'artistId': artistid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_artist_id_by_name(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is a simplified version of the search endpoint that gets the ID of an artist by name."
    name: Artist name.
        
    """
    url = f"https://spotify-scraper.p.rapidapi.com/v1/artist/search"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_genre_contents(genreid: str, offset: int=None, limit: int=None, sublimit: int=None, region: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists sub-genres, playlists, podcasts, albums, etc. of a genre. Pagination scraping is supported."
    genreid: Genre ID.
        offset: The number of items omitted before the results. Defaults to `0`.
        limit: The max number of items returned per request. Defaults to `50`.
        sublimit: The max number of items listed in each sub-genre. Defaults to `10`.
        region: Region code (ISO 3166 alpha-2) for localized results. Defaults to `US`. Unsupported code will **fallback** to `US`.
        
    """
    url = f"https://spotify-scraper.p.rapidapi.com/v1/genre/contents"
    querystring = {'genreId': genreid, }
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if sublimit:
        querystring['sublimit'] = sublimit
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_weekly_top_albums(date: str=None, region: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists weekly top albums at any time and in any country."
    date: Date in `YYYY-MM-DD` format. Leave blank to get the latest chart.
        region: `global` or region code (ISO 3166 alpha-2, e.g., `US`). Defaults to `global`.
        
    """
    url = f"https://spotify-scraper.p.rapidapi.com/v1/chart/albums/top"
    querystring = {}
    if date:
        querystring['date'] = date
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_top_artists(type: str=None, date: str=None, region: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists weekly or daily top artists at any time and in any country."
    type: Time span. Defaults to `weekly`.
        date: Date in `YYYY-MM-DD` format. Leave blank to get the latest chart.
        region: `global` or region code (ISO 3166 alpha-2, e.g., `US`). Defaults to `global`.
        
    """
    url = f"https://spotify-scraper.p.rapidapi.com/v1/chart/artists/top"
    querystring = {}
    if type:
        querystring['type'] = type
    if date:
        querystring['date'] = date
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_daily_viral_tracks(date: str=None, region: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists daily viral tracks at any time and in any country."
    date: Date in `YYYY-MM-DD` format. Leave blank to get the latest chart.
        region: `global` or region code (ISO 3166 alpha-2, e.g., `US`). Defaults to `global`.
        
    """
    url = f"https://spotify-scraper.p.rapidapi.com/v1/chart/tracks/viral"
    querystring = {}
    if date:
        querystring['date'] = date
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_top_tracks(region: str=None, type: str=None, date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists weekly or daily top tracks at any time and in any country."
    region: `global` or region code (ISO 3166 alpha-2, e.g., `US`). Defaults to `global`.
        type: Time span. Defaults to `weekly`.
        date: Date in `YYYY-MM-DD` format. Leave blank to get the latest chart.
        
    """
    url = f"https://spotify-scraper.p.rapidapi.com/v1/chart/tracks/top"
    querystring = {}
    if region:
        querystring['region'] = region
    if type:
        querystring['type'] = type
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_podcast_episodes(showid: str, limit: int=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists all episodes of a podcast."
    showid: Show ID.
        limit: The max number of items returned per request. Defaults to `100`.
        offset: The number of items omitted before the results. Defaults to `0`.
        
    """
    url = f"https://spotify-scraper.p.rapidapi.com/v1/show/episodes"
    querystring = {'showId': showid, }
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_podcast_metadata(showid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint fetches full details of a podcast."
    showid: Show ID.
        
    """
    url = f"https://spotify-scraper.p.rapidapi.com/v1/show/metadata"
    querystring = {'showId': showid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_episode_downloadable_audio(episodeid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint fetches full details of an episode. Downloadable URL of full audio is available for **some** of the episodes."
    episodeid: Episode ID.
        
    """
    url = f"https://spotify-scraper.p.rapidapi.com/v1/episode/details"
    querystring = {'episodeId': episodeid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_user_followers(userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists part of a user's followers."
    userid: User ID.
        
    """
    url = f"https://spotify-scraper.p.rapidapi.com/v1/user/followers"
    querystring = {'userId': userid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_playlist_tracks_and_episodes(playlistid: str, offset: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists tracks and episodes in a playlist."
    playlistid: Playlist ID.
        offset: The number of items omitted before the results. Defaults to `0`.
        limit: The max number of items returned per request. Defaults to `100`.
        
    """
    url = f"https://spotify-scraper.p.rapidapi.com/v1/playlist/contents"
    querystring = {'playlistId': playlistid, }
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_album_tracks(albumid: str, limit: int=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists tracks of an album. Pagination scraping is supported."
    albumid: Album ID.
        limit: The max number of items returned per request. Defaults to `300`.
        offset: The number of items omitted before the results. Defaults to `0`.
        
    """
    url = f"https://spotify-scraper.p.rapidapi.com/v1/album/tracks"
    querystring = {'albumId': albumid, }
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_album_metadata(albumid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint fetches full details of an album."
    albumid: Album ID.
        
    """
    url = f"https://spotify-scraper.p.rapidapi.com/v1/album/metadata"
    querystring = {'albumId': albumid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_home_page_overview(region: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint fetches trending information displayed on the Spotify home page as an incognito visitor."
    region: Region code (ISO 3166 alpha-2) for localized results. Defaults to `US`. Unsupported code will **fallback** to `US`.
        
    """
    url = f"https://spotify-scraper.p.rapidapi.com/v1/home"
    querystring = {}
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

