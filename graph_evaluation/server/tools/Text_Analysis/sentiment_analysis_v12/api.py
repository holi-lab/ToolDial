import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def text_analysis(text: str, output: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Analyzes text using  VADER (Valence Aware Dictionary and sEntiment Reasoner)
		
		Provide  a string and the API will output combined, positive, neutral, and negative scores based on the output specified."
    
    """
    url = f"https://sentiment-analysis14.p.rapidapi.com/sentiment/"
    querystring = {'text': text, }
    if output:
        querystring['output'] = output
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sentiment-analysis14.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

