import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def video_downloader(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "FreeVideoDownloader is a powerful video downloader tool designed for websites, allowing seamless integration of video downloading capabilities. It provides an easy-to-use API that enables website owners to offer their visitors the ability to download videos from various platforms without leaving the site. With FreeVideoDownloader, users can effortlessly save videos in different formats, enhancing their browsing experience and content accessibility."
    
    """
    url = f"https://video-downloader15.p.rapidapi.com/json"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "video-downloader15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

