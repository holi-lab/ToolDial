import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ideapin_download_pinterest_reels(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a url to download the ideapins in original hd quality by passing pinterest ideapin address"
    
    """
    url = f"https://pinterest-downloader-download-image-video-and-pinterest-ideapin.p.rapidapi.com/ideapin/{url}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinterest-downloader-download-image-video-and-pinterest-ideapin.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_download(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a url to download the video in original hd quality by passing pinterest video pin address"
    
    """
    url = f"https://pinterest-downloader-download-image-video-and-pinterest-ideapin.p.rapidapi.com/video/{url}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinterest-downloader-download-image-video-and-pinterest-ideapin.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def image_download(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a url to download the image in original hd quality by passing pinterest image pin address"
    
    """
    url = f"https://pinterest-downloader-download-image-video-and-pinterest-ideapin.p.rapidapi.com/image/{url}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinterest-downloader-download-image-video-and-pinterest-ideapin.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

