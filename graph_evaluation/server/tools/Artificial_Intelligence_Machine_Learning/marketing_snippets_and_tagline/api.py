import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_snippets_and_keywords(prompt: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "AI generated marketing snippets and keywords"
    
    """
    url = f"https://marketing-snippets-and-tagline.p.rapidapi.com/generate_snippets_and_keywords"
    querystring = {'prompt': prompt, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "marketing-snippets-and-tagline.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

