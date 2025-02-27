import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def sentiment_analysis(text: str='Renewable energy in Australia is a growing industry which is good', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint takes a GET request with a string as a parameter and returns the sentiment scores (polarity and subjectivity) of the text as well as the 'Full Text', 'Language', 'Entities', 'Keywords', 'Sentiment Label' and 'Subjectivity Label'."
    
    """
    url = f"https://text-sentiment-api2.p.rapidapi.com/sentiment"
    querystring = {}
    if text:
        querystring['text'] = text
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "text-sentiment-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

