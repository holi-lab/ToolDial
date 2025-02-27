import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def translate(to: str, is_from: str, text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Converting written text or spoken words from one language to another, making it understandable to people who speak different languages."
    to: Language - **ISO-639-1 code**
        from: Language - **ISO-639-1 code**
        
    """
    url = f"https://ai-translation1.p.rapidapi.com/translate"
    querystring = {'to': to, 'from': is_from, 'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ai-translation1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

