import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def video_thumbnail_extractor(videourl: str, percentage: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Extractor thumbnail image from mp4 video"
    videourl: The full URL link of the mp4 video
        percentage: The specific time between 0% - 99% of video duration where the thumbnail image will be extracted. Default value is 50%.
        
    """
    url = f"https://video-thumbnail-extractor.p.rapidapi.com/api/v1/rapid/capivideothumbnailextractor"
    querystring = {'videoUrl': videourl, }
    if percentage:
        querystring['percentage'] = percentage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "video-thumbnail-extractor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_thumbnail_extractor_copy(videourl: str, percentage: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Extractor thumbnail image from mp4 video"
    videourl: The full URL link of the mp4 video
        percentage: The specific time between 0% - 99% of video duration where the thumbnail image will be extracted. Default value is 50%.
        
    """
    url = f"https://video-thumbnail-extractor.p.rapidapi.com/api/v1/rapid/capivideothumbnailextractor"
    querystring = {'videoUrl': videourl, }
    if percentage:
        querystring['percentage'] = percentage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "video-thumbnail-extractor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

