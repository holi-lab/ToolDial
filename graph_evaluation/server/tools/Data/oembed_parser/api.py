import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_oembed_parse(url: str, maxheight: str='400', theme: str='dark', lang: str='en', maxwidth: int=600, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Extract oEmbed data from given URL."
    
    """
    url = f"https://oembed-parser.p.rapidapi.com/oembed/parse"
    querystring = {'url': url, }
    if maxheight:
        querystring['maxheight'] = maxheight
    if theme:
        querystring['theme'] = theme
    if lang:
        querystring['lang'] = lang
    if maxwidth:
        querystring['maxwidth'] = maxwidth
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "oembed-parser.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

