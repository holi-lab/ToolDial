import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def subtitle(videoid: str, responsetype: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch YouTube video caption/subtitle by channel Id
		
		Default format will be JSON
		
		You can get Caption Subtitles .srt format as well"
    responsetype: Convert response into .json or .srt format
        
    """
    url = f"https://youtube-all-in-one.p.rapidapi.com/captions"
    querystring = {'videoID': videoid, 'responseType': responsetype, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-all-in-one.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channel_info(channelid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "YouTube channel information by channel id"
    
    """
    url = f"https://youtube-all-in-one.p.rapidapi.com/channel"
    querystring = {'channelId': channelid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-all-in-one.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_video(keywords: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "YouTube search videos by keywords"
    
    """
    url = f"https://youtube-all-in-one.p.rapidapi.com/search_videos"
    querystring = {'keywords': keywords, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-all-in-one.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_channels(keywords: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "YouTube search channels by keywords"
    
    """
    url = f"https://youtube-all-in-one.p.rapidapi.com/search_channel"
    querystring = {'keywords': keywords, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-all-in-one.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def comments(videoid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "YouTube video comments list by channel id"
    
    """
    url = f"https://youtube-all-in-one.p.rapidapi.com/comments"
    querystring = {'videoId': videoid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-all-in-one.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

