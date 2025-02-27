import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def video_details(videoid: str, geo: str='GB', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns:
		1. Details about the video;
		2. A list of similar videos along with some basic details about them. The continuation of the list can be obtained in the *Videos/Similar Videos* endpoint by passing the `continuation` parameter obtained from this endpoint."
    
    """
    url = f"https://fastytapi.p.rapidapi.com/ytapi/videoDetails"
    querystring = {'videoId': videoid, }
    if geo:
        querystring['geo'] = geo
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fastytapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channel_details(channelid: str, geo: str='GB', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns channel details, including: description, number of videos/views/subscribers, thumbnails, etc.. Se example on the right for a full list."
    channelid: 24-characters-long channel id
        
    """
    url = f"https://fastytapi.p.rapidapi.com/ytapi/channelDetails"
    querystring = {'channelId': channelid, }
    if geo:
        querystring['geo'] = geo
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fastytapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def similar_videos(videoid: str='qebMrMt4240', continuation: str=None, geo: str='GB', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "If  `videoId` is provided (and `continuation` is not provided) returns a list of similar videos.
		
		If  `continuation` is provided (and `videoId` is not provided) returns the continuation of the list of similar videos. The `continuation` parameter can be obtained from this endpoint, when passing only the `videoId` in a first request, or from the *Videos/Video Details* endpoint."
    videoid: (required if `continuation` is not provided)
        continuation: (required if `videoId` is not provided)
        
    """
    url = f"https://fastytapi.p.rapidapi.com/ytapi/similarVideos"
    querystring = {}
    if videoid:
        querystring['videoId'] = videoid
    if continuation:
        querystring['continuation'] = continuation
    if geo:
        querystring['geo'] = geo
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fastytapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def similar_channels(channelid: str, geo: str='GB', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of related channels for a given channel. Each similar channel comes with the same information that the *Channel Details*  endpoint returns, plus the extra `similarityScore` field indicating how similar the channel is. Score is arbitrary, higher means more similar.
		
		**Note:** this endpoint requires heavier computations, response time might take ~5 seconds."
    
    """
    url = f"https://fastytapi.p.rapidapi.com/ytapi/similarChannels"
    querystring = {'channelId': channelid, }
    if geo:
        querystring['geo'] = geo
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fastytapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_videos(videostype: str, channelid: str, continuation: str=None, geo: str='GB', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of videos, streams (a.k.a. lives) and shorts along with some basic information about each of them. See one of the items in the example response on the right. Results are ordered by "latest".
		
		For more details about each video use the *Videos/Video Details* endpoint."
    videostype: The type of videos: Shorts/Streams/Videos
        continuation: Allows to retrieve the continuation of the list. Pass the value received from a previous request to this endpoint. NB the `videosType` parameter is still **required**!
        
    """
    url = f"https://fastytapi.p.rapidapi.com/ytapi/listVideos"
    querystring = {'videosType': videostype, 'channelId': channelid, }
    if continuation:
        querystring['continuation'] = continuation
    if geo:
        querystring['geo'] = geo
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fastytapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def youtube_search(resultstype: str, sortby: str, query: str, uploaddate: str='thisMonth', geo: str='GB', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search YouTube for channels or videos based on a given query and the given filters.
		
		See parameters below for available options."
    
    """
    url = f"https://fastytapi.p.rapidapi.com/ytapi/search"
    querystring = {'resultsType': resultstype, 'sortBy': sortby, 'query': query, }
    if uploaddate:
        querystring['uploadDate'] = uploaddate
    if geo:
        querystring['geo'] = geo
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fastytapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

