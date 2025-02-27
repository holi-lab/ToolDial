import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def youtube_downloader(url: str='www.youtube.com/watch?v=8FOg2nYHXb8', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "send a GET request to this endpoint with url query and the link as parameter for the  downloadable video what you desire."
    
    """
    url = f"https://youtube-downloader-api-best-resolution.p.rapidapi.com/convert"
    querystring = {}
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-downloader-api-best-resolution.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

