import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def politicians(offset: int=None, limit: int=None, active: bool=None, province: str=None, municipality: str=None, level_of_government: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "fetch a list of politicians data"
    
    """
    url = f"https://goverlytics1.p.rapidapi.com/politicians"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if active:
        querystring['active'] = active
    if province:
        querystring['province'] = province
    if municipality:
        querystring['municipality'] = municipality
    if level_of_government:
        querystring['level_of_government'] = level_of_government
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "goverlytics1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def politician_goverlytics_id(goverlytics_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get 1 politician given their 'goverlytics_id'."
    
    """
    url = f"https://goverlytics1.p.rapidapi.com/politician/{goverlytics_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "goverlytics1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def records(limit: int=None, offset: int=None, substring: str=None, speaker_id: str=None, topic: str=None, speaker_party: str=None, level_of_government: str=None, region: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetches records data"
    
    """
    url = f"https://goverlytics1.p.rapidapi.com/records"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if substring:
        querystring['substring'] = substring
    if speaker_id:
        querystring['speaker_id'] = speaker_id
    if topic:
        querystring['topic'] = topic
    if speaker_party:
        querystring['speaker_party'] = speaker_party
    if level_of_government:
        querystring['level_of_government'] = level_of_government
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "goverlytics1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

