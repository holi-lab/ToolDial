import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def answer(context: str, prompt: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Context**[string]: plaintext or  *domain-url*
		**Prompt**[string]: plaintext
		
		`Note`: url will detect automatically any context beginning with 'https'. Capped at 5000 tokens."
    
    """
    url = f"https://openmanual.p.rapidapi.com/answer"
    querystring = {'context': context, 'prompt': prompt, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openmanual.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

