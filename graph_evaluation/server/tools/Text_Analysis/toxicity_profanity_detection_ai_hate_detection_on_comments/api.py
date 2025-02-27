import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_toxicity_score(text: str, language: str='english', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the TOXICITY, SEVERE_TOXICITY, IDENTITY_ATTACK, INSULT, PROFANITY, & THREAT score for the given text. The score ranges from 0 to 1.
		
		You can provide a language (recommended), or the language will be auto-detected.
		 
		You can provide the following languages, either the language itself or the language code: English (en), Spanish (es), Arabic (ar), Chinese (zh), Russian (ru), French (fr), German (de), Japanese (ja), Portuguese (pt), Italian (it), Korean (ko), Dutch (nl), Swedish (sv), Polish (pl), Indonesian (id), Czech (cs), Hindi (hi), Hinglish (hi-Latn)"
    
    """
    url = f"https://toxicity-profanity-detection-ai-hate-detection-on-comments3.p.rapidapi.com/moderation/"
    querystring = {'text': text, }
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "toxicity-profanity-detection-ai-hate-detection-on-comments3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

