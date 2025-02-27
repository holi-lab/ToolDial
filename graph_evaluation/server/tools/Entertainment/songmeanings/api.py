import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def comments_get(artist_name: str, lyric_title: str, sm_lid: str=None, parent_id: str=None, type: str='all', page: str='1', page_size: str='25', page_order: str='date', page_sort: str='asc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get comments from our database for a specific song."
    artist_name: The artist name
        lyric_title: The song, track or lyric title
        sm_lid: If SM lyric_id is known, then this will bypass artist_name and lyric_title
        parent_id: When specified, method will return any replies for parent comment.
        type: Set what type of comments to return
        page: Starting page of comments
        page_size: Set how many comments returned per page
        page_order: Order by a specific field (date or rating)
        page_sort: Sort by ascending or descending (asc or desc)
        
    """
    url = f"https://songmeanings.p.rapidapi.com/"
    querystring = {'artist_name': artist_name, 'lyric_title': lyric_title, }
    if sm_lid:
        querystring['sm_lid'] = sm_lid
    if parent_id:
        querystring['parent_id'] = parent_id
    if type:
        querystring['type'] = type
    if page:
        querystring['page'] = page
    if page_size:
        querystring['page_size'] = page_size
    if page_order:
        querystring['page_order'] = page_order
    if page_sort:
        querystring['page_sort'] = page_sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "songmeanings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def artists_search(q: str, page: str='1', page_size: str='25', matchmode: str='extended', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search the SongMeanings database for artist names and aliases."
    q: Query string
        page: Starting page of comments
        page_size: Set how many comments returned per page
        matchmode: Configure how matching occurs (see further parameter values)
        
    """
    url = f"https://songmeanings.p.rapidapi.com/"
    querystring = {'q': q, }
    if page:
        querystring['page'] = page
    if page_size:
        querystring['page_size'] = page_size
    if matchmode:
        querystring['matchmode'] = matchmode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "songmeanings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lyrics_get(sm_lid: str, lyric_title: str, artist_name: str, format: str, spotify_id: str=None, count: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves lyrics and lyric related data for a specific song."
    sm_lid: SongMeanings Lyric ID. When specified, this parameter will bypass any other lyric identification parameters.
        lyric_title: Song or track title.
        artist_name: Artist name.
        format: Decide the output type (either xml or json); xml is default.
        spotify_id: If specified, a match will attempt to be made via spotify_id. Please note that if parameter is used, the API will also require lyric_title and artist_name as a backup attempt.
        count: This parameter is used to track, and report, any offline cached views of lyrics.
        
    """
    url = f"https://songmeanings.p.rapidapi.com/"
    querystring = {'sm_lid': sm_lid, 'lyric_title': lyric_title, 'artist_name': artist_name, 'format': format, }
    if spotify_id:
        querystring['spotify_id'] = spotify_id
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "songmeanings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def songs_search(q: str, sm_aid: str=None, q_artist: str=None, index: str=None, page: str='1', page_size: str='25', matchmode: str='extended', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    q: Query string
        sm_aid: SongMeapings Artist ID. When defined, engine will search query for any song titles found under the artist specified.
        q_artist: Artist name. When defined and Artist ID acquired, engine will search query for any song titles found under the artist specified. If both sm_aid and q_artist are defined, sm_aid will always supercede.
        index: Additional indexes to search in addition to song titles (see further parameter values)
        page: Starting page of comments
        page_size: Set how many comments returned per page
        matchmode: Configure how matching occurs (see further parameter values)
        
    """
    url = f"https://songmeanings.p.rapidapi.com/"
    querystring = {'q': q, }
    if sm_aid:
        querystring['sm_aid'] = sm_aid
    if q_artist:
        querystring['q_artist'] = q_artist
    if index:
        querystring['index'] = index
    if page:
        querystring['page'] = page
    if page_size:
        querystring['page_size'] = page_size
    if matchmode:
        querystring['matchmode'] = matchmode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "songmeanings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def users_favorite_get(sm_uid: str, type: str='lyrics', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a user’s favorites by specified type."
    sm_uid: User ID to grab favorites for
        type: Type of favorites to retrieve (artists, lyrics)
        
    """
    url = f"https://songmeanings.p.rapidapi.com/"
    querystring = {'sm_uid': sm_uid, }
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "songmeanings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def users_password_reset(sm_emai: str, sm_uid: str=None, sm_username: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Allows for a user to reset their password. When initiated, method will send the user an email with a link to click. The link will bring the user to desktop SongMeanings where the user will complete the password reset."
    sm_emai: The user’s email address on the account to be reset.
        sm_uid: SM User ID of the user initiating the password reset request.
        sm_username: SM Username of the user initiating the password reset request.
        
    """
    url = f"https://songmeanings.p.rapidapi.com/"
    querystring = {'sm_emai': sm_emai, }
    if sm_uid:
        querystring['sm_uid'] = sm_uid
    if sm_username:
        querystring['sm_username'] = sm_username
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "songmeanings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

