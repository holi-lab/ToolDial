import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def view_song(country: str, song_preview_kbps: str, https_only: str, wmid: str, songid: str, lang: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "View song"
    
    """
    url = f"https://view-song.p.rapidapi.com/web-fcgi-bin/web_get_songinfo"
    querystring = {'country': country, 'song_preview_kbps': song_preview_kbps, 'https_only': https_only, 'wmid': wmid, 'songid': songid, 'lang': lang, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "view-song.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def view_song_copy(https_only: str, wmid: str, songid: str, s: str, song_preview_kbps: str, lang: str, country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "View song"
    
    """
    url = f"https://view-song.p.rapidapi.com/web-fcgi-bin/web_get_songinfo"
    querystring = {'https_only': https_only, 'wmid': wmid, 'songid': songid, 's': s, 'song_preview_kbps': song_preview_kbps, 'lang': lang, 'country': country, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "view-song.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

