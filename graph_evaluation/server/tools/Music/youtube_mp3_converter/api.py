import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_download_url_mp3_mp4(lang: str, is_id: str, action: str, widget: str, format: str='mp3', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Download URL"
    lang: The language
        id: The youtube video id
        format: mp3 or mp4 
        
    """
    url = f"https://youtube-mp3-converter.p.rapidapi.com/service/run?"
    querystring = {'lang': lang, 'id': is_id, 'action': action, 'widget': widget, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-mp3-converter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

