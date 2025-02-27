import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def text2wav_memory_output(phrase: str, voice: str='female', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this Endpoint to Convert Text to  Audio Memory /Browser Object"
    phrase: This is text phrase to be converted into speech in memory, limited to **400** characters.

        voice: Specifies the voice of speaker.** It can be either male or female. **
        
    """
    url = f"https://text2wav1.p.rapidapi.com/textToSpeechMem"
    querystring = {'phrase': phrase, }
    if voice:
        querystring['voice'] = voice
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "text2wav1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def text2wav_file_output(phrase: str, voice: str='male', filetype: str='wav', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this Endpoint to Convert Text to Downloadable Audio File(wav,mp3,ogg)"
    phrase: This is text phrase to be converted into speech, limited to **400** characters.
        voice: Specifies the voice of speaker.** It can be either male or female. **
        filetype: The is the filetype of the output speech file. Valid values are -  **wav or mp3 or ogg**
        
    """
    url = f"https://text2wav1.p.rapidapi.com/textToSpeechFile"
    querystring = {'phrase': phrase, }
    if voice:
        querystring['voice'] = voice
    if filetype:
        querystring['filetype'] = filetype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "text2wav1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

