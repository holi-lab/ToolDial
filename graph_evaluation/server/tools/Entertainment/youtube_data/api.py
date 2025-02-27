import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def auto_complete(q: str, hl: str='en', gl: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "auto-complete"
    q: query text
        
    """
    url = f"https://youtube-data8.p.rapidapi.com/auto-complete/"
    querystring = {'q': q, }
    if hl:
        querystring['hl'] = hl
    if gl:
        querystring['gl'] = gl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-data8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_streaming_data(is_id: str='VyHV0BRtdxo', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "video-streaming-data"
    is_id: Video ID
        
    """
    url = f"https://youtube-data8.p.rapidapi.com/video/streaming-data/"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-data8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_related_contents(hl: str='en', is_id: str='kJQP7kiw5Fk', gl: str='US', cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "video-related-contents"
    is_id: Video ID
        cursor: Cursor token
        
    """
    url = f"https://youtube-data8.p.rapidapi.com/video/related-contents/"
    querystring = {}
    if hl:
        querystring['hl'] = hl
    if is_id:
        querystring['id'] = is_id
    if gl:
        querystring['gl'] = gl
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-data8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def community_post_comments(cursor: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "community-post-comments"
    cursor: Cursor token

You can get it from the Community Post Details endpoint.
        
    """
    url = f"https://youtube-data8.p.rapidapi.com/community-post/comments/"
    querystring = {'cursor': cursor, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-data8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def community_post_details(is_id: str='UgkxCWeKPiIOLsnh_5a0MPHWCmYgbhifgwIZ', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "community-post-details"
    is_id: Community post ID
        
    """
    url = f"https://youtube-data8.p.rapidapi.com/community-post/details/"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-data8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(cursor: str=None, hl: str='en', q: str='movie', gl: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search"
    
    """
    url = f"https://youtube-data8.p.rapidapi.com/search/"
    querystring = {}
    if cursor:
        querystring['cursor'] = cursor
    if hl:
        querystring['hl'] = hl
    if q:
        querystring['q'] = q
    if gl:
        querystring['gl'] = gl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-data8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channel_community(cursor: str=None, is_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "channel-community"
    cursor: Cursor token
        is_id: Channel ID
        
    """
    url = f"https://youtube-data8.p.rapidapi.com/channel/community/"
    querystring = {}
    if cursor:
        querystring['cursor'] = cursor
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-data8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_comments(cursor: str=None, gl: str='US', is_id: str='kJQP7kiw5Fk', hl: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "video-comments"
    cursor: Cursor token
        is_id: Video ID
        
    """
    url = f"https://youtube-data8.p.rapidapi.com/video/comments/"
    querystring = {}
    if cursor:
        querystring['cursor'] = cursor
    if gl:
        querystring['gl'] = gl
    if is_id:
        querystring['id'] = is_id
    if hl:
        querystring['hl'] = hl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-data8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playlist_videos(gl: str='US', cursor: str=None, hl: str='en', is_id: str='PLcirGkCPmbmFeQ1sm4wFciF03D_EroIfr', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "playlist-videos"
    cursor: Cursor token
        is_id: Playlist ID
        
    """
    url = f"https://youtube-data8.p.rapidapi.com/playlist/videos/"
    querystring = {}
    if gl:
        querystring['gl'] = gl
    if cursor:
        querystring['cursor'] = cursor
    if hl:
        querystring['hl'] = hl
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-data8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playlist_details(is_id: str='PLcirGkCPmbmFeQ1sm4wFciF03D_EroIfr', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "playlist-details"
    is_id: Playlist ID
        
    """
    url = f"https://youtube-data8.p.rapidapi.com/playlist/details/"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-data8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_details(is_id: str, hl: str='en', gl: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "video-details"
    id: Video ID
        
    """
    url = f"https://youtube-data8.p.rapidapi.com/video/details/"
    querystring = {'id': is_id, }
    if hl:
        querystring['hl'] = hl
    if gl:
        querystring['gl'] = gl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-data8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channel_channels(is_id: str='UC-lHJZR3Gqxm24_Vd_AJ5Yw', gl: str='US', cursor: str=None, hl: str='en', filter: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "channel-channels"
    is_id: Channel ID
        filter: Filter key or token, default: all_collections

Keys you can enter:

all_collections: Returns channel collections
subscriptions: Returns subscribed channels
or custom collection token
        
    """
    url = f"https://youtube-data8.p.rapidapi.com/channel/channels/"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    if gl:
        querystring['gl'] = gl
    if cursor:
        querystring['cursor'] = cursor
    if hl:
        querystring['hl'] = hl
    if filter:
        querystring['filter'] = filter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-data8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channel_playlists(hl: str='en', filter: str=None, cursor: str=None, is_id: str='UC-lHJZR3Gqxm24_Vd_AJ5Yw', gl: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "channel-playlists"
    filter: Filter key or token, default: all_collections

Keys you can enter:

all_collections: Returns playlist collections
created_playlists_newest: Returns created playlists (by newest)
created_playlists_last_video_added: Returns created playlists (by last video added)
saved_playlists: Returns saved playlists
or custom collection token
        is_id: Channel ID
        
    """
    url = f"https://youtube-data8.p.rapidapi.com/channel/playlists/"
    querystring = {}
    if hl:
        querystring['hl'] = hl
    if filter:
        querystring['filter'] = filter
    if cursor:
        querystring['cursor'] = cursor
    if is_id:
        querystring['id'] = is_id
    if gl:
        querystring['gl'] = gl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-data8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channel_videos(is_id: str='UC-lHJZR3Gqxm24_Vd_AJ5Yw', hl: str='en', filter: str=None, cursor: str=None, gl: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "channel-videos"
    filter: Filter key, default: videos_latest

Keys you can enter:

videos_latest: Returns videos (by latest)
streams_latest: Returns live streams (by latest)
shorts_latest: Returns short videos (by latest)
live_now: Returns current live streams
        
    """
    url = f"https://youtube-data8.p.rapidapi.com/channel/videos/"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    if hl:
        querystring['hl'] = hl
    if filter:
        querystring['filter'] = filter
    if cursor:
        querystring['cursor'] = cursor
    if gl:
        querystring['gl'] = gl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-data8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channel_details(is_id: str, hl: str='en', gl: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "channel-details"
    
    """
    url = f"https://youtube-data8.p.rapidapi.com/channel/details/"
    querystring = {'id': is_id, }
    if hl:
        querystring['hl'] = hl
    if gl:
        querystring['gl'] = gl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-data8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

