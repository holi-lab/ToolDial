import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_keyword_search_volume_not_working_as_of_now(keyword: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "*This is recommended to use when you need to search volume only because in this endpoint we avoid calculation of other matrices.
		*
		Get the Search Volume of any keyword."
    
    """
    url = f"https://keysuggest-keyword-data.p.rapidapi.com/get_keyword_search_volume"
    querystring = {'keyword': keyword, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "keysuggest-keyword-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_keyword_data(keyword: str, related_kw: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about keywords including Search Volume,  CPC, KD, PD, Intent, and Parent Topic."
    
    """
    url = f"https://keysuggest-keyword-data.p.rapidapi.com/get_keyword_data"
    querystring = {'keyword': keyword, }
    if related_kw:
        querystring['related_kw'] = related_kw
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "keysuggest-keyword-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

