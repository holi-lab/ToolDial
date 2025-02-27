import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def wrt_transformer(text: str, topics: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the text and set of possible topics separated by a comma.
		
		Returns the ranking of topics from most relevant to least relevant."
    
    """
    url = f"https://walnut-topic.p.rapidapi.com/wrt_transformer"
    querystring = {'text': text, 'topics': topics, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "walnut-topic.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

