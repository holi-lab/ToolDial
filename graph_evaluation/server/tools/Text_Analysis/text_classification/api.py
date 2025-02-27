import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def classify_get(text: str, title: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Classify text into product categories or contact us to customize and use your own category sets.  Enter some text to find its related product categories:"
    text: Enter some text to find related categories.
        title: Enter title of text (optional).
        
    """
    url = f"https://twinword-text-classification.p.rapidapi.com/classify/"
    querystring = {'text': text, }
    if title:
        querystring['title'] = title
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twinword-text-classification.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

