import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_videos_videoid_chapters(videoid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://api-video4.p.rapidapi.com/videos/{videoid}/chapters"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-video4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_videos_videoid_captions(videoid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://api-video4.p.rapidapi.com/videos/{videoid}/captions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-video4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_analytics_live_streams_livestreamid(livestreamid: str, currentpage: int=1, period: str=None, pagesize: int=25, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    currentpage: Number of the page to display
        period: Period must have one of the following formats: 

- For a day : 2018-01-01,
- For a week: 2018-W01, 
- For a month: 2018-01
- For a year: 2018

For a range period: 
-  Date range: 2018-01-01/2018-01-15

        pagesize: Expected number of items to display on the page. Might be lower on the last page
        
    """
    url = f"https://api-video4.p.rapidapi.com/analytics/live-streams/{livestreamid}"
    querystring = {}
    if currentpage:
        querystring['currentPage'] = currentpage
    if period:
        querystring['period'] = period
    if pagesize:
        querystring['pageSize'] = pagesize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-video4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_analytics_videos_videoid(videoid: str, metadata: str='[]', period: str=None, currentpage: int=1, pagesize: int=25, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    metadata: Metadata and Dynamic Metadata filter.
(Dynamic metadata filter are available for  Business plans only)
        period: Period must have one of the following formats: 

- For a day : 2018-01-01,
- For a week: 2018-W01, 
- For a month: 2018-01
- For a year: 2018

For a range period: 
-  Date range: 2018-01-01/2018-01-15

        currentpage: Number of the page to display
        pagesize: Expected number of items to display on the page. Might be lower on the last page
        
    """
    url = f"https://api-video4.p.rapidapi.com/analytics/videos/{videoid}"
    querystring = {}
    if metadata:
        querystring['metadata'] = metadata
    if period:
        querystring['period'] = period
    if currentpage:
        querystring['currentPage'] = currentpage
    if pagesize:
        querystring['pageSize'] = pagesize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-video4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_analytics_sessions_sessionid_events(sessionid: str, currentpage: int=1, pagesize: int=25, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Useful to track and measure video's engagement."
    currentpage: Number of the page to display
        pagesize: Expected number of items to display on the page. Might be lower on the last page
        
    """
    url = f"https://api-video4.p.rapidapi.com/analytics/sessions/{sessionid}/events"
    querystring = {}
    if currentpage:
        querystring['currentPage'] = currentpage
    if pagesize:
        querystring['pageSize'] = pagesize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-video4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_videos_videoid_chapters_language(videoid: str, language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    language: A valid BCP 47 language representation
        
    """
    url = f"https://api-video4.p.rapidapi.com/videos/{videoid}/chapters/{language}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-video4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_players(pagesize: int=25, sortby: str='createdAt', currentpage: int=1, sortorder: str='asc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    pagesize: Expected number of items to display on the page. Might be lower on the last page
        currentpage: Number of the page to display
        sortorder: Allowed: asc, desc
        
    """
    url = f"https://api-video4.p.rapidapi.com/players"
    querystring = {}
    if pagesize:
        querystring['pageSize'] = pagesize
    if sortby:
        querystring['sortBy'] = sortby
    if currentpage:
        querystring['currentPage'] = currentpage
    if sortorder:
        querystring['sortOrder'] = sortorder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-video4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_live_streams_livestreamid(livestreamid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://api-video4.p.rapidapi.com/live-streams/{livestreamid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-video4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_videos(currentpage: int=1, livestreamid: str=None, sortorder: str=None, tags: str=None, pagesize: int=25, sortby: str=None, description: str=None, title: str=None, metadata: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    currentpage: Search results page. Minimum value: 1
        sortorder: Allowed: asc, desc
        pagesize: Results per page. Allowed values 1-100, default is 25.
        sortby: Allowed: publishedAt, title
        metadata: metadata[foo]=bar
        
    """
    url = f"https://api-video4.p.rapidapi.com/videos"
    querystring = {}
    if currentpage:
        querystring['currentPage'] = currentpage
    if livestreamid:
        querystring['liveStreamId'] = livestreamid
    if sortorder:
        querystring['sortOrder'] = sortorder
    if tags:
        querystring['tags'] = tags
    if pagesize:
        querystring['pageSize'] = pagesize
    if sortby:
        querystring['sortBy'] = sortby
    if description:
        querystring['description'] = description
    if title:
        querystring['title'] = title
    if metadata:
        querystring['metadata'] = metadata
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-video4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_video_status(videoid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API provides upload status & encoding status to determine when the video is uploaded or ready to playback.
		
		Once encoding is completed, the response also lists the available stream qualities."
    
    """
    url = f"https://api-video4.p.rapidapi.com/videos/{videoid}/status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-video4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_live_streams(currentpage: int=1, sortby: str=None, streamkey: str=None, name: str=None, pagesize: int=25, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    currentpage: Number of the page to display
        sortby: Allowed: createdAt, publishedAt, name
        pagesize: Expected number of items to display on the page. Might be lower on the last page
        
    """
    url = f"https://api-video4.p.rapidapi.com/live-streams"
    querystring = {}
    if currentpage:
        querystring['currentPage'] = currentpage
    if sortby:
        querystring['sortBy'] = sortby
    if streamkey:
        querystring['streamKey'] = streamkey
    if name:
        querystring['name'] = name
    if pagesize:
        querystring['pageSize'] = pagesize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-video4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_players_playerid(playerid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://api-video4.p.rapidapi.com/players/{playerid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-video4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_video(videoid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This call provides the same JSON information provided on video creation. For private videos, it will generate a unique token url."
    
    """
    url = f"https://api-video4.p.rapidapi.com/videos/{videoid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-video4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_account(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://api-video4.p.rapidapi.com/account"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-video4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_videos_videoid_captions_language(videoid: str, language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    language: A valid BCP 47 language representation
        
    """
    url = f"https://api-video4.p.rapidapi.com/videos/{videoid}/captions/{language}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-video4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

