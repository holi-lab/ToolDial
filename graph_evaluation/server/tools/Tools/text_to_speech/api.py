import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_language(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get list of available language for text to speech"
    
    """
    url = f"https://text-to-speech27.p.rapidapi.com/speech/lang"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "text-to-speech27.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def send_text_to_speech_stream(text: str, lang: str='en-us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "text to speech 
		
		text = the text you want to speak
		
		language = default en , get the list of supported language for get /tts/speech/lang
		
		
		```
		console.log(response);
		        audio.pause();
		        audio.src = URL.createObjectURL(response.data);
		        audio.play();
		
		```"
    lang: default is en-us
list can be get by /speech/lang

e.g. zh-tw
        
    """
    url = f"https://text-to-speech27.p.rapidapi.com/speech"
    querystring = {'text': text, }
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "text-to-speech27.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

