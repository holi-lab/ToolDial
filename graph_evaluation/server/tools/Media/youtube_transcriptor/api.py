import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def transcript(video_id: str, lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Transcript videos"
    video_id: Example : youtube.com/watch?v=`0-fD8SDEvrR`
        lang: **auto detect** language if none was passed.
        
    """
    url = f"https://youtube-transcriptor.p.rapidapi.com/transcript"
    querystring = {'video_id': video_id, }
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-transcriptor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

