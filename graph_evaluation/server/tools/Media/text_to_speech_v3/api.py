import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_tts_voices(language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the available voices"
    language: Returns the names of voices that are available for the given language
        
    """
    url = f"https://tts.p.rapidapi.com/v1/tts/voices"
    querystring = {'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tts.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_tts_languages(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the available languages"
    
    """
    url = f"https://tts.p.rapidapi.com/v1/tts/languages"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tts.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_tts(text: str, format: str, voice: str='The name of the voice you wish to use.', language: str='The language you wish to use (default is usenglish)', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Converts text into high quality speech."
    text: Whatever text you want to be converted into audio
        format: wav, mp3, and ogg file formats are supported
        voice: See /v1/tts/voices for details on how to query the installed voices
        language: See /v1/tts/languages to query the available languages
        
    """
    url = f"https://tts.p.rapidapi.com/v1/tts"
    querystring = {'text': text, 'format': format, }
    if voice:
        querystring['voice'] = voice
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tts.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

