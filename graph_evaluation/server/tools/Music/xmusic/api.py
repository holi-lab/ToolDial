import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def xmusic_song_info(lang: str, country: str, songid: str, song_preview_kbps: str, https_only: str, wmid: str='161445361', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Test apis xmusic song info"
    
    """
    url = f"https://xmusic.p.rapidapi.com/web-fcgi-bin/web_get_songinfo"
    querystring = {'lang': lang, 'country': country, 'songid': songid, 'song_preview_kbps': song_preview_kbps, 'https_only': https_only, }
    if wmid:
        querystring['wmid'] = wmid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "xmusic.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

