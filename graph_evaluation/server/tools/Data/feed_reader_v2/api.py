import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def load(url: str, dateculture: str=None, offset: int=0, maxcount: int=None, dateformat: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    url: Feed Url
        dateculture: Date Format Language/Culture for Feed Publish Date formatting. eg: en-US or fr-FR or tr-TR etc.
        offset: Offset count for pagination. Defaut=0, Use offset (skip) & maxCount(take) for pagination
        maxcount: Maximum Feed Item Count
        dateformat: Date Format for Feed Publish Date. eg: MM/dd/yyyy
        
    """
    url = f"https://feed-reader3.p.rapidapi.com/load"
    querystring = {'url': url, }
    if dateculture:
        querystring['dateCulture'] = dateculture
    if offset:
        querystring['offset'] = offset
    if maxcount:
        querystring['maxCount'] = maxcount
    if dateformat:
        querystring['dateFormat'] = dateformat
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "feed-reader3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

