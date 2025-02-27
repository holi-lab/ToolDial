import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def analyse_text_sentiment(text: str, lang: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns a JSON containing:
		- Sentiment type (positive, negative or neutro)
		- Score"
    
    """
    url = f"https://multilingual-text-sentiment-analysis.p.rapidapi.com/analysetext"
    querystring = {'text': text, 'lang': lang, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "multilingual-text-sentiment-analysis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

