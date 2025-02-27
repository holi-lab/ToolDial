import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def emotion_detection(text: str="I'm so happy", toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The API takes a GET request with a 'text' parameter, which represents the input text to be analyzed.
		Based on the sentiment score, the API returns an emotion label that corresponds to the sentiment of the input text. The emotion labels range from "terrified" for a very negative sentiment to "overjoyed" for a very positive sentiment, with many other emotions in between."
    
    """
    url = f"https://emotion-detection-api.p.rapidapi.com/analyze_sentiment"
    querystring = {}
    if text:
        querystring['text'] = text
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emotion-detection-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

