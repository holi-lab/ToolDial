import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def find_quotes(author: str='Barack Obama', topic: str='tax rates', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find quotes from people about certain topics."
    author: The author of the quote or empty if the author does not matter.
        topic: The text that must appear in the quote. Leave that empty if you want all quotes from the given author.
        
    """
    url = f"https://webknox-entities.p.rapidapi.com/entities/quotes"
    querystring = {}
    if author:
        querystring['author'] = author
    if topic:
        querystring['topic'] = topic
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webknox-entities.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def detect_entities(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Detect entity mentions in a given text."
    text: The text in which entities should be detected.
        
    """
    url = f"https://webknox-entities.p.rapidapi.com/text/entities"
    querystring = {'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webknox-entities.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

