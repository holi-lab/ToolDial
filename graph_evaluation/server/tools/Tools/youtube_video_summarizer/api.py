import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_youtube_summarizevideofromcache(videourl: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the video summary of a given YouTube video that has already been summarized. The summary is provided in markdown format."
    videourl: The URL of the YouTube video to get from the summary cache.
        
    """
    url = f"https://youtube-video-summarizer1.p.rapidapi.com/v1/youtube/summarizeVideoFromCache"
    querystring = {'videoURL': videourl, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-video-summarizer1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_youtube_summarizevideowithtoken(videourl: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Create a video summary of a given YouTube video. The summary is provided in markdown format."
    videourl: The URL of the YouTube video to summarize.
        
    """
    url = f"https://youtube-video-summarizer1.p.rapidapi.com/v1/youtube/summarizeVideoWithToken"
    querystring = {'videoURL': videourl, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-video-summarizer1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

