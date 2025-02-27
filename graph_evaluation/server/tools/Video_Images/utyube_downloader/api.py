import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def utube_downloader_api_documenttion(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Utube-downloader is a free API to download youtube video from a link in different qualities, You can send request with GET method very easily and it will return you a json object and in json object first key contains the length of arrays and seocnd key videos contains the array of urls."
    
    """
    url = f"https://utyube-downloader.p.rapidapi.com/fetch?url=YOUR_VIDEO_LINK"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "utyube-downloader.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

