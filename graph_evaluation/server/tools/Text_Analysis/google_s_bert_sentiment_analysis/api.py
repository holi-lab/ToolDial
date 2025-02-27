import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def sentiment_analysis(text: str, lang: str='id', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Multi-lingual Sentiment Analysis parameter {lang} is optional, we can detect the language but you can define it for better result and accurate"
    
    """
    url = f"https://googles-bert-sentiment-analysis.p.rapidapi.com/sentiment"
    querystring = {'text': text, }
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "googles-bert-sentiment-analysis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

