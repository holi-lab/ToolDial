import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def text_similarity_get(text1: str, text2: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Evaluate the similarity of two words, sentences, or paragraphs."
    text1: Input the first text to compare.
        text2: Input a second text to compare its similarity with the first text.
        
    """
    url = f"https://twinword-text-similarity-v1.p.rapidapi.com/similarity/"
    querystring = {'text1': text1, 'text2': text2, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twinword-text-similarity-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

