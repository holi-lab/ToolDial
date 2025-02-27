import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def api_key_validation(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used only to validate API Keys, it simply returns a status of 200 and a message of 'OK'. 
		The use of this endpoint is Free, and is useful for server-to-server FREE API validation."
    
    """
    url = f"https://adcopy-ai-google-ads-ai-text-generation.p.rapidapi.com/v1/seo/validateApi"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "adcopy-ai-google-ads-ai-text-generation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

