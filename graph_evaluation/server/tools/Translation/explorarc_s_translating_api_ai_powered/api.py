import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def translator(lang: str='spanish', query: str='eamless translation between multiple languagesS', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "ExplorArc's Translating API is an **AI-powered tool** that enables seamless translation between multiple languages, providing accurate and reliable results."
    
    """
    url = f"https://explorarcs-translating-api-ai-powered.p.rapidapi.com/api/translator/translate"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    if query:
        querystring['query'] = query
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "explorarcs-translating-api-ai-powered.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

