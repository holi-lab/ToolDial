import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def enter_url_for_photos_reels_igtv_videos(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Enter URL for Photos / Reels / IGTV / Videos**
		*Dont Put Story Url here*
		
		Example:-
		*https://www.instagram.com/p/Cultg9GuVh_/*"
    
    """
    url = f"https://instagram-downloader-download-videos-reels-stories-2023.p.rapidapi.com/"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-downloader-download-videos-reels-stories-2023.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

