import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def download_youtube_to_mp3(url: str, quality: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get download links in: 320Kbps, 256Kbps, 192, 128 & 64Kbps. (Specify the quality in the field)"
    quality: Select the audio quality of the output mp3. Allowed values are:
320
192
256
128
64
        
    """
    url = f"https://youtube-mp3-download-highest-quality1.p.rapidapi.com/ytmp3/ytmp3/custom/"
    querystring = {'url': url, 'quality': quality, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-mp3-download-highest-quality1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

