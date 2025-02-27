import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def detect_27_emotions(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Detect up to 27 emotions in any text.
		Simply enter your string in the "text" parameter, and the endpoint will return a JSON object as a result."
    text: The text to be analyzed
        
    """
    url = f"https://advanced-emotions-detection-advemotions.p.rapidapi.com/getemotionsnon"
    querystring = {'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "advanced-emotions-detection-advemotions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

