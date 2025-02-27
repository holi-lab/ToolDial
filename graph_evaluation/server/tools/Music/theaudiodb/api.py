import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def return_all_album_details_from_artist_name(s: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return all Album details from artist name"
    s: artist name
        
    """
    url = f"https://theaudiodb.p.rapidapi.com/searchalbum.php"
    querystring = {'s': s, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theaudiodb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def return_all_albums_for_an_artist_using_known_tadb_artist_id(i: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return All Albums for an Artist using known TADB_Artist_ID"
    i: TheAudioDB Artist ID
        
    """
    url = f"https://theaudiodb.p.rapidapi.com/album.php"
    querystring = {'i': i, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theaudiodb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def return_all_tracks_for_album_from_known_tadb_album_id(m: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return All Tracks for Album from known TADB_Album_ID"
    m: TheAudioDB Album ID
        
    """
    url = f"https://theaudiodb.p.rapidapi.com/track.php"
    querystring = {'m': m, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theaudiodb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def return_artist_details_from_artist_name(s: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return Artist details from artist name"
    s: artist name
        
    """
    url = f"https://theaudiodb.p.rapidapi.com/search.php"
    querystring = {'s': s, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theaudiodb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def return_discography_for_an_artist_with_album_names_and_year_only(s: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return Discography for an Artist with Album names and year only"
    s: artist
        
    """
    url = f"https://theaudiodb.p.rapidapi.com/discography.php"
    querystring = {'s': s, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theaudiodb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def return_individual_album_info_using_known_tadb_album_id(m: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return individual Album info using known TADB_Album_ID"
    m: TheAudioDB Album ID
        
    """
    url = f"https://theaudiodb.p.rapidapi.com/album.php"
    querystring = {'m': m, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theaudiodb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def return_individual_artist_details_using_known_tadb_artist_id_theaudiodb_artist_id(i: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return individual Artist details using known TADB_Artist_ID (TheAudioDB Artist ID)"
    i: TheAudioDB Artist ID
        
    """
    url = f"https://theaudiodb.p.rapidapi.com/artist.php"
    querystring = {'i': i, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theaudiodb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def return_single_album_details_from_album_name(a: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return single album details from album name"
    a: album name
        
    """
    url = f"https://theaudiodb.p.rapidapi.com/searchalbum.php"
    querystring = {'a': a, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theaudiodb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def return_single_album_details_from_artist_album_name(s: str, a: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return single album details from artist + album name"
    s: artist name
        a: album name
        
    """
    url = f"https://theaudiodb.p.rapidapi.com/searchalbum.php"
    querystring = {'s': s, 'a': a, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theaudiodb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def return_track_details_from_artist_track_name(t: str, s: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return track details from artist/track name"
    t: track name
        s: artist
        
    """
    url = f"https://theaudiodb.p.rapidapi.com/searchtrack.php"
    querystring = {'t': t, 's': s, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theaudiodb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def return_individual_track_info_using_a_known_tadb_track_id(h: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return individual track info using a known TADB_Track_ID"
    h: TheAudioDB Track ID
        
    """
    url = f"https://theaudiodb.p.rapidapi.com/track.php"
    querystring = {'h': h, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theaudiodb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def return_all_the_music_videos_for_a_known_tadb_artist_id(i: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return all the Music videos for a known TADB_Artist_ID"
    i: TheAudioDB Artist ID
        
    """
    url = f"https://theaudiodb.p.rapidapi.com/mvid.php"
    querystring = {'i': i, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theaudiodb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mbid_return_individual_artist_info_using_a_known_musicbrainz_artist_id(i: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return individual Artist info using a known MusicBrainz_Artist_ID"
    i: Music Brainz ID (https://musicbrainz.org/)
        
    """
    url = f"https://theaudiodb.p.rapidapi.com/artist-mb.php"
    querystring = {'i': i, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theaudiodb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mbid_return_individual_album_info_using_a_known_musicbrainz_release_group_id(i: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return individual Album info using a known MusicBrainz_Release-Group_ID"
    i: Music Brainz ID (https://musicbrainz.org/)
        
    """
    url = f"https://theaudiodb.p.rapidapi.com/album-mb.php"
    querystring = {'i': i, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theaudiodb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mbid_return_individual_track_info_using_a_known_musicbrainz_recording_id(i: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return individual track info using a known MusicBrainz_Recording_ID"
    i: Music Brainz Recording ID
        
    """
    url = f"https://theaudiodb.p.rapidapi.com/track-mb.php"
    querystring = {'i': i, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theaudiodb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mbid_return_all_the_music_videos_for_a_known_music_brainz_id(i: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return all the Music videos for a known Music_Brainz_ID"
    i: Music Brainz ID (https://musicbrainz.org/)
        
    """
    url = f"https://theaudiodb.p.rapidapi.com/mvid-mb.php"
    querystring = {'i': i, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theaudiodb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def return_the_top_10_most_loved_tracks_for_an_artist_name(s: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return The top 10 Most Loved tracks for an Artist Name"
    s: artist name
        
    """
    url = f"https://theaudiodb.p.rapidapi.com/track-top10.php"
    querystring = {'s': s, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theaudiodb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def return_the_top_10_most_loved_tracks_for_an_artist_music_brainz_id(s: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return The top 10 Most Loved tracks for an Artist Music Brainz ID"
    s: Music Brainz Artist ID
        
    """
    url = f"https://theaudiodb.p.rapidapi.com/track-top10-mb.php"
    querystring = {'s': s, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theaudiodb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def return_the_top_50_most_loved_tracks_of_alltime(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return The top 50 Most Loved Tracks of Alltime"
    
    """
    url = f"https://theaudiodb.p.rapidapi.com/mostloved.php?format=track"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theaudiodb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def return_the_top_10_most_loved_albums_of_alltime(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return The top 10 Most Loved Albums of Alltime"
    
    """
    url = f"https://theaudiodb.p.rapidapi.com/mostloved.php?format=album"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theaudiodb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def return_the_most_recent_track_playlists(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return The most recent Track playlists"
    
    """
    url = f"https://theaudiodb.p.rapidapi.com/playlist.php?format=track"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theaudiodb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def return_the_most_recent_album_playlists(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return The most recent Album playlists"
    
    """
    url = f"https://theaudiodb.p.rapidapi.com/playlist.php?format=album"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theaudiodb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def return_the_most_recent_user_album_playlists(format: str, user: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return The most recent user Album playlists"
    format: "track" or "album"
        user: TheAudioDB User
        
    """
    url = f"https://theaudiodb.p.rapidapi.com/playlist.php"
    querystring = {'format': format, 'user': user, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theaudiodb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_playlists(s: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search playlists"
    s: Search Playlists
        
    """
    url = f"https://theaudiodb.p.rapidapi.com/playlist.php"
    querystring = {'s': s, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theaudiodb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lookup_playlists(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lookup playlists"
    id: playlist id
        
    """
    url = f"https://theaudiodb.p.rapidapi.com/playlist.php"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theaudiodb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_every_rating_made_by_a_user_album(user: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List every rating made by a user"
    user: username
        
    """
    url = f"https://theaudiodb.p.rapidapi.com/ratings-album.php"
    querystring = {'user': user, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theaudiodb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_every_rating_made_by_a_user_track(user: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List every rating made by a user"
    user: username
        
    """
    url = f"https://theaudiodb.p.rapidapi.com/ratings-track.php"
    querystring = {'user': user, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theaudiodb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def current_trending_music(country: str, type: str, format: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Current Trending Music"
    country: us, gb, de, fr, it
        type: itunes (only for now but will extend in future to more sources)
        format: albums, singles
        
    """
    url = f"https://theaudiodb.p.rapidapi.com/trending.php"
    querystring = {'country': country, 'type': type, 'format': format, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theaudiodb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

