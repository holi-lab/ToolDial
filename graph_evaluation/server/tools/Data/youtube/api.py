import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(hl: str='en', gl: str='US', cursor: str=None, q: str='despacito', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search"
    cursor: Cursor token
        q: Search query
        
    """
    url = f"https://youtube138.p.rapidapi.com/search/"
    querystring = {}
    if hl:
        querystring['hl'] = hl
    if gl:
        querystring['gl'] = gl
    if cursor:
        querystring['cursor'] = cursor
    if q:
        querystring['q'] = q
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube138.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def auto_complete(q: str, hl: str='en', gl: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Auto Complete"
    q: Query for suggestions
        
    """
    url = f"https://youtube138.p.rapidapi.com/auto-complete/"
    querystring = {'q': q, }
    if hl:
        querystring['hl'] = hl
    if gl:
        querystring['gl'] = gl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube138.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def home(hl: str='en', cursor: str=None, gl: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Home"
    cursor: Cursor token
        
    """
    url = f"https://youtube138.p.rapidapi.com/home/"
    querystring = {}
    if hl:
        querystring['hl'] = hl
    if cursor:
        querystring['cursor'] = cursor
    if gl:
        querystring['gl'] = gl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube138.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channel_channels(cursor: str=None, gl: str='US', is_id: str='UCJ5v_MCY6GNUBTO8-D3XoAg', filter: str=None, hl: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Channel Channels"
    is_id: Channel ID
        filter: Filter key or token, default: `all_collections`

Keys you can enter:

- `all_collections`: Returns channel collections
- `subscriptions`: Returns subscribed channels
- or custom collection token
        
    """
    url = f"https://youtube138.p.rapidapi.com/channel/channels/"
    querystring = {}
    if cursor:
        querystring['cursor'] = cursor
    if gl:
        querystring['gl'] = gl
    if is_id:
        querystring['id'] = is_id
    if filter:
        querystring['filter'] = filter
    if hl:
        querystring['hl'] = hl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube138.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channel_community(is_id: str='UCJ5v_MCY6GNUBTO8-D3XoAg', cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Channel Community"
    is_id: Channel ID
        cursor: Cursor token
        
    """
    url = f"https://youtube138.p.rapidapi.com/channel/community/"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube138.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channel_playlists(filter: str=None, is_id: str='UCJ5v_MCY6GNUBTO8-D3XoAg', gl: str='US', hl: str='en', cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Channel Playlists"
    filter: Filter key or token, default: `all_collections`

Keys you can enter:

- `all_collections`: Returns playlist collections
- `created_playlists_newest`: Returns created playlists (by newest)
- `created_playlists_last_video_added`: Returns created playlists (by last video added)
- `saved_playlists`: Returns saved playlists
- or custom collection token
        is_id: Channel ID
        cursor: Cursor token
        
    """
    url = f"https://youtube138.p.rapidapi.com/channel/playlists/"
    querystring = {}
    if filter:
        querystring['filter'] = filter
    if is_id:
        querystring['id'] = is_id
    if gl:
        querystring['gl'] = gl
    if hl:
        querystring['hl'] = hl
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube138.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channel_videos(hl: str='en', cursor: str=None, gl: str='US', is_id: str='UCJ5v_MCY6GNUBTO8-D3XoAg', filter: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Channel Videos"
    cursor: Cursor token
        is_id: Channel ID
        filter: Filter key, default: `videos_latest`

Keys you can enter:

- `videos_latest`: Returns videos (by latest)
- `streams_latest`: Returns live streams (by latest)
- `shorts_latest`: Returns short videos (by latest)
- `live_now`: Returns current live streams
        
    """
    url = f"https://youtube138.p.rapidapi.com/channel/videos/"
    querystring = {}
    if hl:
        querystring['hl'] = hl
    if cursor:
        querystring['cursor'] = cursor
    if gl:
        querystring['gl'] = gl
    if is_id:
        querystring['id'] = is_id
    if filter:
        querystring['filter'] = filter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube138.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channel_details(is_id: str, gl: str='US', hl: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Channel Details"
    id: Channel ID

Starts with the `UC` prefix
        
    """
    url = f"https://youtube138.p.rapidapi.com/channel/details/"
    querystring = {'id': is_id, }
    if gl:
        querystring['gl'] = gl
    if hl:
        querystring['hl'] = hl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube138.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channel_search(q: str='john cena', hl: str='en', cursor: str=None, is_id: str='UCJ5v_MCY6GNUBTO8-D3XoAg', gl: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Channel Search"
    q: Search query
        cursor: Cursor token
        is_id: Channel ID
        
    """
    url = f"https://youtube138.p.rapidapi.com/channel/search/"
    querystring = {}
    if q:
        querystring['q'] = q
    if hl:
        querystring['hl'] = hl
    if cursor:
        querystring['cursor'] = cursor
    if is_id:
        querystring['id'] = is_id
    if gl:
        querystring['gl'] = gl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube138.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def community_post_comments(cursor: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Community Post Comments"
    cursor: Cursor token

*You can get it from the `Community Post Details` endpoint.*
        
    """
    url = f"https://youtube138.p.rapidapi.com/community-post/comments/"
    querystring = {'cursor': cursor, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube138.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def community_post_details(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Community Post Details"
    id: Community post ID
        
    """
    url = f"https://youtube138.p.rapidapi.com/community-post/details/"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube138.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_streaming_data(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Video Streaming Data"
    id: Video ID
        
    """
    url = f"https://youtube138.p.rapidapi.com/video/streaming-data/"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube138.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_details(is_id: str, hl: str='en', gl: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Video Details"
    id: Video ID
        
    """
    url = f"https://youtube138.p.rapidapi.com/video/details/"
    querystring = {'id': is_id, }
    if hl:
        querystring['hl'] = hl
    if gl:
        querystring['gl'] = gl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube138.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playlist_videos(cursor: str=None, hl: str='en', gl: str='US', is_id: str='PLcirGkCPmbmFeQ1sm4wFciF03D_EroIfr', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Playlist Videos"
    cursor: Cursor token
        is_id: Playlist ID
        
    """
    url = f"https://youtube138.p.rapidapi.com/playlist/videos/"
    querystring = {}
    if cursor:
        querystring['cursor'] = cursor
    if hl:
        querystring['hl'] = hl
    if gl:
        querystring['gl'] = gl
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube138.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playlist_details(is_id: str, gl: str='US', hl: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Playlist Details"
    id: Playlist ID
        
    """
    url = f"https://youtube138.p.rapidapi.com/playlist/details/"
    querystring = {'id': is_id, }
    if gl:
        querystring['gl'] = gl
    if hl:
        querystring['hl'] = hl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube138.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_related_contents(cursor: str=None, is_id: str='kJQP7kiw5Fk', gl: str='US', hl: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Video Related Contents"
    cursor: Cursor token
        is_id: Video ID
        
    """
    url = f"https://youtube138.p.rapidapi.com/video/related-contents/"
    querystring = {}
    if cursor:
        querystring['cursor'] = cursor
    if is_id:
        querystring['id'] = is_id
    if gl:
        querystring['gl'] = gl
    if hl:
        querystring['hl'] = hl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube138.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_comments(gl: str='US', is_id: str='kJQP7kiw5Fk', cursor: str=None, hl: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Video Comments"
    is_id: Video ID
        cursor: Cursor token
        
    """
    url = f"https://youtube138.p.rapidapi.com/video/comments/"
    querystring = {}
    if gl:
        querystring['gl'] = gl
    if is_id:
        querystring['id'] = is_id
    if cursor:
        querystring['cursor'] = cursor
    if hl:
        querystring['hl'] = hl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube138.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

