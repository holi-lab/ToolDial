import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def list_voices(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of famous voices you can use to generate text. The list includes the {voice_id} parameter which will be used in the /generateMessage endpoint to select the voice. The list also contains the speaker/character of the voice."
    
    """
    url = f"https://ai-voice-text-to-speech-with-celebrities.p.rapidapi.com/listVoices"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ai-voice-text-to-speech-with-celebrities.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

