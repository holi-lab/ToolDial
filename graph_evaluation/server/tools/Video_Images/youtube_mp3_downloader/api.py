import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def download_mp3_custom_audio_quality(url: str, quality: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Download YouTube to Mp3 in various audio quality: 320 Kbps, 192Kbps, 256Kbps, 128Kbps & 64Kbps. (Specify the quality in the field)"
    quality: Select the audio quality of the output mp3. Allowed values are:
320
192
256
128
64
        
    """
    url = f"https://youtube-mp3-downloader2.p.rapidapi.com/ytmp3/ytmp3/custom/"
    querystring = {'url': url, 'quality': quality, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-mp3-downloader2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download_youtube_mp3(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Update: 17 May 2023: API is fixed & it is now 100% working. Converts any YouTube video into Mp3 downloadable files. No Ads. No wild redirects."
    
    """
    url = f"https://youtube-mp3-downloader2.p.rapidapi.com/ytmp3/ytmp3/"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-mp3-downloader2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

