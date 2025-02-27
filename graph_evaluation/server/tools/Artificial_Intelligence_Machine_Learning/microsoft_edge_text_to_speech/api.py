import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getvoiceslist(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Voices list for the param "voice_name" of the function "GetDownloadUrl""
    
    """
    url = f"https://microsoft-edge-text-to-speech.p.rapidapi.com/TTS/VoicesList"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "microsoft-edge-text-to-speech.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdownloadurl(text: str, voice_name: str='en-US-AriaNeural', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "request with a long text and get the mp3 download URL created by the TTS engine.
		
		Supporting lots of languages with different voices, such as French, Spanish, Portuguese, Japanese, Korean, Chinese, Polish, Hindi and so on
		
		params:
		text - STRING : the text to be transformed to speeches.
		voice_name - STRING: the voice and language for the speeches."
    
    """
    url = f"https://microsoft-edge-text-to-speech.p.rapidapi.com/TTS/EdgeTTS"
    querystring = {'text': text, }
    if voice_name:
        querystring['voice_name'] = voice_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "microsoft-edge-text-to-speech.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

