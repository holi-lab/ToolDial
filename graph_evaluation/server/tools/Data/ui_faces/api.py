import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ui_faces(x_api_key: str, limit: str=None, offset: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate UI Faces"
    x_api_key: API key can be obtained from uifaces.co/api-docs
        limit: Limit the number of results
        offset: Offset where to start the sequence from
        
    """
    url = f"https://mightyalex-ui-faces-v1.p.rapidapi.com/"
    querystring = {'X-API-KEY': x_api_key, }
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mightyalex-ui-faces-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

