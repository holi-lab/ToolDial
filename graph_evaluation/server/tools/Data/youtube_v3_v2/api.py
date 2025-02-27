import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def channel_videos(channelid: str, part: str, maxresults: str='50', pagetoken: str=None, order: str='date', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get channel videos"
    
    """
    url = f"https://youtube-v31.p.rapidapi.com/search"
    querystring = {'channelId': channelid, 'part': part, }
    if maxresults:
        querystring['maxResults'] = maxresults
    if pagetoken:
        querystring['pageToken'] = pagetoken
    if order:
        querystring['order'] = order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playlist_videos(part: str, playlistid: str, pagetoken: str=None, maxresults: str='50', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get playlist videos"
    
    """
    url = f"https://youtube-v31.p.rapidapi.com/playlistItems"
    querystring = {'part': part, 'playlistId': playlistid, }
    if pagetoken:
        querystring['pageToken'] = pagetoken
    if maxresults:
        querystring['maxResults'] = maxresults
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playlist_details(is_id: str, part: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get playlist details"
    
    """
    url = f"https://youtube-v31.p.rapidapi.com/playlists"
    querystring = {'id': is_id, 'part': part, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_comments(videoid: str, part: str, maxresults: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get YouTube video comments."
    
    """
    url = f"https://youtube-v31.p.rapidapi.com/commentThreads"
    querystring = {'videoId': videoid, 'part': part, 'maxResults': maxresults, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channel_details(is_id: str, part: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get channel details"
    
    """
    url = f"https://youtube-v31.p.rapidapi.com/channels"
    querystring = {'id': is_id, 'part': part, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_details(is_id: str, part: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get YouTube video details.
		
		Note:
		**topicDetails** part is not enabled. If you want this part to be included in the API response then please contact us."
    
    """
    url = f"https://youtube-v31.p.rapidapi.com/videos"
    querystring = {'id': is_id, 'part': part, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def suggested_videos(type: str, part: str, relatedtovideoid: str, maxresults: int=50, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Similar videos"
    
    """
    url = f"https://youtube-v31.p.rapidapi.com/search"
    querystring = {'type': type, 'part': part, 'relatedToVideoId': relatedtovideoid, }
    if maxresults:
        querystring['maxResults'] = maxresults
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def comment_info(part: str, is_id: str, maxresults: int=None, parentid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get comments info."
    
    """
    url = f"https://youtube-v31.p.rapidapi.com/comments"
    querystring = {'part': part, 'id': is_id, }
    if maxresults:
        querystring['maxResults'] = maxresults
    if parentid:
        querystring['parentId'] = parentid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def captions_list(part: str, videoid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of caption tracks that are associated with a specified video"
    
    """
    url = f"https://youtube-v31.p.rapidapi.com/captions"
    querystring = {'part': part, 'videoId': videoid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(part: str, q: str, pagetoken: str=None, order: str='date', regioncode: str='US', maxresults: int=50, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search from YouTube"
    
    """
    url = f"https://youtube-v31.p.rapidapi.com/search"
    querystring = {'part': part, 'q': q, }
    if pagetoken:
        querystring['pageToken'] = pagetoken
    if order:
        querystring['order'] = order
    if regioncode:
        querystring['regionCode'] = regioncode
    if maxresults:
        querystring['maxResults'] = maxresults
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

