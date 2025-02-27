import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def umuziki(umuziki: str='music streaming', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Stream free albums and hits, find a song, discover music, and download songs and podcasts with the Umuziki free streaming and music player for Africa."
    
    """
    url = f"https://umuziki.p.rapidapi.com/"
    querystring = {}
    if umuziki:
        querystring['umuziki'] = umuziki
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "umuziki.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

