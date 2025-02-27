import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def easytorrents(type: str, name: str, language: str, quality: str='1080p', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Example Request"
    type: Type (movie, show, anime)
        name: Movie Title
        language: Language (en or fr)
        quality: quality (ex. 720p, HD, Web RIP) can be empty for it to find the best available.
        
    """
    url = f"https://easytorrents1.p.rapidapi.com/"
    querystring = {'type': type, 'name': name, 'language': language, }
    if quality:
        querystring['quality'] = quality
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "easytorrents1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

