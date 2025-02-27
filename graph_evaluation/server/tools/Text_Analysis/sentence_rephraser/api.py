import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def rewrite_sentence(sourcetext: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint automatically rewrites sentences, phrases or paragraphs and removes plagiarism."
    
    """
    url = f"https://sentence-rephraser.p.rapidapi.com/rewrite/"
    querystring = {'sourceText': sourcetext, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sentence-rephraser.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

