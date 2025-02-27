import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def title_url(url: str, title: str='Call of Duty : Modern Warfare 2 Remastered - All Weapon Reload Animations in 4 Minutes', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "fetches YouTube direct download link and video title"
    
    """
    url = f"https://youtube-to-mp4.p.rapidapi.com/url=&title"
    querystring = {'url': url, }
    if title:
        querystring['title'] = title
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-to-mp4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

