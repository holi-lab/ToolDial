import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def translatetext(tolanguage: str, text: str, fromlanguage: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter the text to be translated."
    tolanguage: Enter the language you want to translate the text to. (Use the language codes in About/Read Me)
        text: Enter the text to be translated
        fromlanguage: Enter the original language of text. (Use the language codes in About/Read Me)
        
    """
    url = f"https://google-translate-text.p.rapidapi.com/getTranslation"
    querystring = {'toLanguage': tolanguage, 'text': text, }
    if fromlanguage:
        querystring['fromLanguage'] = fromlanguage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-translate-text.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def texttospeech(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "convert a given text into speech. Note the audio file is available for 5 minutes maximum at our servers. Download as soon as possible
		
		We are fixing some few bugs in this endpoint."
    
    """
    url = f"https://google-translate-text.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-translate-text.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def supportedlanguages(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of supported languages"
    
    """
    url = f"https://google-translate-text.p.rapidapi.com/supportedLanguages"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-translate-text.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def detectlanguage(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get language from text"
    
    """
    url = f"https://google-translate-text.p.rapidapi.com/detectLanguage"
    querystring = {'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-translate-text.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

