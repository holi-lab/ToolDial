import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def youtube_keyword_search(q: str, gl: str='us', hl: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "YouTube Keyword Search"
    q: Search Query
        gl: Country Code (Alpha-2)

Default: `us`
        hl: Search Language Code (ISO 639-1)

Default: `en`
        
    """
    url = f"https://youtube-keyword-search.p.rapidapi.com/"
    querystring = {'q': q, }
    if gl:
        querystring['gl'] = gl
    if hl:
        querystring['hl'] = hl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-keyword-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

