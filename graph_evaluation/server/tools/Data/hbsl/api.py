import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def bible(verse: str=None, chapter: str=None, abbreviation: str=None, bibleid: str=None, is_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Holy Bible Sign Language"
    
    """
    url = f"https://hbsl.p.rapidapi.com/bible"
    querystring = {}
    if verse:
        querystring['verse'] = verse
    if chapter:
        querystring['chapter'] = chapter
    if abbreviation:
        querystring['abbreviation'] = abbreviation
    if bibleid:
        querystring['bibleid'] = bibleid
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hbsl.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

