import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def channel_search_continuation(query: str, continuation_token: str, channel_id: str, lang: str='en', country: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gets search the next list of search results in a given Youtube channel using the channel ID"
    
    """
    url = f"https://youtube-v2.p.rapidapi.com/channel/search/continuation"
    querystring = {'query': query, 'continuation_token': continuation_token, 'channel_id': channel_id, }
    if lang:
        querystring['lang'] = lang
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channel_videos(channel_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the latest videos published by a Youtube channel"
    
    """
    url = f"https://youtube-v2.p.rapidapi.com/channel/videos"
    querystring = {'channel_id': channel_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def youtube_channel_id(channel_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint get the channel ID of a Youtube Channel given the channel name"
    
    """
    url = f"https://youtube-v2.p.rapidapi.com/channel/id"
    querystring = {'channel_name': channel_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channel_search(query: str, channel_id: str, lang: str='en', country: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gets search results in a given Youtube channel using the channel ID"
    
    """
    url = f"https://youtube-v2.p.rapidapi.com/channel/search"
    querystring = {'query': query, 'channel_id': channel_id, }
    if lang:
        querystring['lang'] = lang
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channel_details(channel_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint get all available details about a given channel ID"
    
    """
    url = f"https://youtube-v2.p.rapidapi.com/channel/details"
    querystring = {'channel_id': channel_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channel_shorts(channel_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the latest shorts published by a Youtube channel"
    
    """
    url = f"https://youtube-v2.p.rapidapi.com/channel/shorts"
    querystring = {'channel_id': channel_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_recommendation(video_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a list of recommended videos based on a given a video ID"
    
    """
    url = f"https://youtube-v2.p.rapidapi.com/video/recommendations"
    querystring = {'video_id': video_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_comments(video_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a list comments under a given Youtube video"
    
    """
    url = f"https://youtube-v2.p.rapidapi.com/video/comments"
    querystring = {'video_id': video_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_subtitles(video_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the available subtitles of a Youtube Video"
    
    """
    url = f"https://youtube-v2.p.rapidapi.com/video/subtitles"
    querystring = {'video_id': video_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_details(video_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all publicly available details about a Youtube Video"
    
    """
    url = f"https://youtube-v2.p.rapidapi.com/video/details"
    querystring = {'video_id': video_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_data(video_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get downloadable links of the Video"
    
    """
    url = f"https://youtube-v2.p.rapidapi.com/video/data"
    querystring = {'video_id': video_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def youtube_search(query: str, lang: str='en', order_by: str='this_month', country: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will a specific number of videos for a specific keyword, note that the maximum is 40 videos per request"
    order_by: Possible values: \\\\\\\"last_hour\\\\\\\", \\\\\\\"today\\\\\\\", \\\\\\\"this_week\\\\\\\", \\\\\\\"this_month\\\\\\\", \\\\\\\"this_year\\\\\\\"
        
    """
    url = f"https://youtube-v2.p.rapidapi.com/search/"
    querystring = {'query': query, }
    if lang:
        querystring['lang'] = lang
    if order_by:
        querystring['order_by'] = order_by
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trending_videos(country: str='us', lang: str='en', section: str='Now', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the list of trending videos given a country"
    section: Possible values: \\\"Now\\\", \\\"Music\\\", \\\"Movies\\\", \\\"Gaming\\\"
        
    """
    url = f"https://youtube-v2.p.rapidapi.com/trending/"
    querystring = {}
    if country:
        querystring['country'] = country
    if lang:
        querystring['lang'] = lang
    if section:
        querystring['section'] = section
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

