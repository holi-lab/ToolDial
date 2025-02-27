import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def givemees(language: str, word: str, has_audio: str=None, translatedto: str='jpn', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for example sentences."
    language: The language of examples you want to get. Must be specified in the ISO 639-2 format (3 letter format, example: eng, jpn).
        word: Specify the word to search. Advanced search options for Tatoeba.org search can be used.
        has_audio: The only possible usage is has&#95;audio=yes. Don't use if you don't need audio of the example sentences. If has&#95;audio=yes the result will include only examples with audio and will include the link to the audio file. If not used, the link to the audio file will not be included in the results.
        translatedto: If used, will give the translations to the specified language in the result. If not, the results will not include translations. Must be specified in the ISO 639-2 format (3 letter format, example: eng, jpn).
        
    """
    url = f"https://givemees.p.rapidapi.com/givemees"
    querystring = {'language': language, 'word': word, }
    if has_audio:
        querystring['has_audio'] = has_audio
    if translatedto:
        querystring['translatedTo'] = translatedto
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "givemees.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

