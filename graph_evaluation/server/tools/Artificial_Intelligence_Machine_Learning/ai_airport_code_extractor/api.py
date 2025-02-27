import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def code_extractor(extractor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will extract airport codes from text."
    extractor: This end point will extract airport codes from text.
        
    """
    url = f"https://ai-airport-code-extractor.p.rapidapi.com/extractor"
    querystring = {}
    if extractor:
        querystring['extractor'] = extractor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ai-airport-code-extractor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

