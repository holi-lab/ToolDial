import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def translate_to_old_english(x_funtranslations_api_secret: str, text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Old English Translator"
    x_funtranslations_api_secret: API Key ( Get yours here : http://funtranslations.com/api/oldenglish )
        text: Text to convert to old English.
        
    """
    url = f"https://orthosie-old-english-translator-v1.p.rapidapi.com/oldenglish.json"
    querystring = {'X-FunTranslations-Api-Secret': x_funtranslations_api_secret, 'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "orthosie-old-english-translator-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

