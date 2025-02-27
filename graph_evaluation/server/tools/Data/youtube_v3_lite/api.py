import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def channel_videos_playlists(channelid: str, part: str, x_cachebypass: str=None, type: str=None, order: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get channel videos or playlists.
		Quota cost is 1."
    channelid: Channel id.
        x_cachebypass: Cache bypass, the value must be 1
Quota Cost +1
        order: Sort parameter:

- `date` [default]
- `viewCount`
        
    """
    url = f"https://youtube-v3-lite.p.rapidapi.com/search"
    querystring = {'channelId': channelid, 'part': part, }
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    if type:
        querystring['type'] = type
    if order:
        querystring['order'] = order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v3-lite.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trending_charts(part: str, regioncode: str, chart: str, x_cachebypass: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Trending Videos list or Chart.
		Quota cost is 1."
    regioncode: ISO 3166-2 country code of the region. Like US (default), UK, CA, IN, etc.
        chart: Chart name.
        x_cachebypass: Cache bypass, the value must be 1
Quota Cost +1
        
    """
    url = f"https://youtube-v3-lite.p.rapidapi.com/videos"
    querystring = {'part': part, 'regionCode': regioncode, 'chart': chart, }
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v3-lite.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(part: str, q: str, x_cachebypass: str=None, videoduration: str=None, videodefinition: str=None, videocaption: str=None, videolicense: str=None, eventtype: str=None, type: str=None, order: str=None, videotype: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get search results.
		Quota cost is 1."
    q: Search query
        x_cachebypass: Cache bypass, the value must be 1
Quota Cost +1
        videoduration: Video duration options:
- `any` [default]
- `short`
- `medium`
- `long`
        videodefinition: Video quality definition options:
- `any` [default]
- `high`
        videocaption: Video captions options:
- `any` [default]
- `closedCaption`
        videolicense: Video license options:
- `any` [default]
- `creativeCommon`
        eventtype: Event type options:
- `any` [default]
- `live`
        type: Type of results:

- `video` [default]
- `playlist`
- `channel`
        order: Sorting order:

- `relevance` [default]
- `date`
- `viewCount`
- `rating`
        videotype: Video type options:
- `any` [default]
- `episode`
- `movie`
        
    """
    url = f"https://youtube-v3-lite.p.rapidapi.com/search"
    querystring = {'part': part, 'q': q, }
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    if videoduration:
        querystring['videoDuration'] = videoduration
    if videodefinition:
        querystring['videoDefinition'] = videodefinition
    if videocaption:
        querystring['videoCaption'] = videocaption
    if videolicense:
        querystring['videoLicense'] = videolicense
    if eventtype:
        querystring['eventType'] = eventtype
    if type:
        querystring['type'] = type
    if order:
        querystring['order'] = order
    if videotype:
        querystring['videoType'] = videotype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v3-lite.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def comments(is_id: str, part: str, x_cachebypass: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get comments list.
		Quota cost is 1."
    id: Video id
        x_cachebypass: Cache bypass, the value must be 1
Quota Cost +1
        
    """
    url = f"https://youtube-v3-lite.p.rapidapi.com/comments"
    querystring = {'id': is_id, 'part': part, }
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v3-lite.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playlist_items(part: str, playlistid: str, x_cachebypass: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get playlist items.
		Quota cost is 1."
    playlistid: Playlist id
        x_cachebypass: Cache bypass, the value must be 1
Quota Cost +1
        
    """
    url = f"https://youtube-v3-lite.p.rapidapi.com/playlistItems"
    querystring = {'part': part, 'playlistId': playlistid, }
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v3-lite.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playlists(part: str, is_id: str, x_cachebypass: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get playlist details.
		Quota cost is 1."
    id: Playlist id
        x_cachebypass: Cache bypass, the value must be 1
Quota Cost +1
        
    """
    url = f"https://youtube-v3-lite.p.rapidapi.com/playlists"
    querystring = {'part': part, 'id': is_id, }
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v3-lite.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def related_videos(part: str, relatedtovideoid: str, type: str, x_cachebypass: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get related videos list.
		Quota cost is 1."
    x_cachebypass: Cache bypass, the value must be 1
Quota Cost +1
        
    """
    url = f"https://youtube-v3-lite.p.rapidapi.com/search"
    querystring = {'part': part, 'relatedToVideoId': relatedtovideoid, 'type': type, }
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v3-lite.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def videos(part: str, is_id: str, x_cachebypass: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get YouTube Video details.
		Quota cost is 1. For statistics part, quota is +1."
    id: YouTube Video id
        x_cachebypass: Cache bypass, the value must be 1
Quota Cost +1
        
    """
    url = f"https://youtube-v3-lite.p.rapidapi.com/videos"
    querystring = {'part': part, 'id': is_id, }
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v3-lite.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channels(x_cachebypass: str=None, part: str='snippet,contentDetails,statistics', is_id: str='UCq-Fj5jknLsUf-MWSy4_brA', forusername: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get channel details.
		Quota cost is 1."
    x_cachebypass: Cache bypass, the value must be 1
Quota Cost +1
        is_id: Channel Id. 
If channel id is not available, then provide `forUsername`
        forusername: Channel username.
If channel username is not available, then provide `id`
        
    """
    url = f"https://youtube-v3-lite.p.rapidapi.com/channels"
    querystring = {}
    if x_cachebypass:
        querystring['X-CACHEBYPASS'] = x_cachebypass
    if part:
        querystring['part'] = part
    if is_id:
        querystring['id'] = is_id
    if forusername:
        querystring['forUsername'] = forusername
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v3-lite.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

