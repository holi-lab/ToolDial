import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def to_get_a_list_all_the_supported_languages_by_this_api_endpoint(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To get a list all the supported languages, execute this API, https://translate.pregnya.com/getSupportedLanguages"
    
    """
    url = f"https://simple-elegant-translation-service.p.rapidapi.com/getSupportedLanguages"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "simple-elegant-translation-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

