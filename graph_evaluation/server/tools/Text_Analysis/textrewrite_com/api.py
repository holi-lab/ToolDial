import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def rewrite_text_get_for_short_texts(text: str, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Just insert your text and have it rewritten instantly."
    text: Text to be rewritten.
        format: Text format (optional). Can be `text` or `html`. Default is `text`.
        
    """
    url = f"https://textrewrite-com.p.rapidapi.com/api.php"
    querystring = {'text': text, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "textrewrite-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

