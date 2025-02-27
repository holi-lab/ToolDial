import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def searchdomain(matchtype: str='prefix', collapse: str='urlkey', url: str='httpsimage.com', fl: str='timestamp:4,original,urlkey', limit: int=100000, filter: str='statuscode:200', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pages-hosted-on-domain.p.rapidapi.com/web/timemap/json"
    querystring = {}
    if matchtype:
        querystring['matchType'] = matchtype
    if collapse:
        querystring['collapse'] = collapse
    if url:
        querystring['url'] = url
    if fl:
        querystring['fl'] = fl
    if limit:
        querystring['limit'] = limit
    if filter:
        querystring['filter'] = filter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pages-hosted-on-domain.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

