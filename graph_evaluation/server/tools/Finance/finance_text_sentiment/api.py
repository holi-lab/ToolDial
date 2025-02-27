import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def negative_sentiment_example(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide a "text" value as any finance text/headline/tweet and get a sentiment with sentiment_score. 
		
		Note: For debugging reasons, the request text is also returned for now, which will be disabled in the future."
    
    """
    url = f"https://finance-text-sentiment.p.rapidapi.com/sentiment_finance"
    querystring = {'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finance-text-sentiment.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def positive_sentiment_example(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide a "text" value as any finance text/headline/tweet and get a sentiment with sentiment_score. 
		
		Note: For debugging reasons, the request text is also returned for now, which will be disabled in the future."
    
    """
    url = f"https://finance-text-sentiment.p.rapidapi.com/sentiment_finance"
    querystring = {'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finance-text-sentiment.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

