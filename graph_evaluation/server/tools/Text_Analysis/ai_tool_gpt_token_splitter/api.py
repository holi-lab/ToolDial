import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def using_path_parameters(text: str, max_size: int=1024, delimiter: str=None, include_variations: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "limited input text size, limited delimiter quantity."
    
    """
    url = f"https://ai-tool-gpt-token-splitter.p.rapidapi.com/api/v1/token-splitter/{text}"
    querystring = {}
    if max_size:
        querystring['max_size'] = max_size
    if delimiter:
        querystring['delimiter'] = delimiter
    if include_variations:
        querystring['include_variations'] = include_variations
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ai-tool-gpt-token-splitter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

