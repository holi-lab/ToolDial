import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def transcribemedia(video_id: str, platform: str='youtube', input_lang: str='en', target_lang: str='fr', streaming_datas: bool=None, model: str='tiny_en', user_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "returns an object containing the timestamps of the translation/transcription as key, and the transcription/translation as value"
    
    """
    url = f"https://media-transcriber.p.rapidapi.com/transcribeMedia"
    querystring = {'video_id': video_id, }
    if platform:
        querystring['platform'] = platform
    if input_lang:
        querystring['input_lang'] = input_lang
    if target_lang:
        querystring['target_lang'] = target_lang
    if streaming_datas:
        querystring['streaming_datas'] = streaming_datas
    if model:
        querystring['model'] = model
    if user_id:
        querystring['user_id'] = user_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "media-transcriber.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getplatforms(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "returns an array with all the supported platforms, with an example link on how to find the user_id parameter and video_id parameter"
    
    """
    url = f"https://media-transcriber.p.rapidapi.com/getPlatforms"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "media-transcriber.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmodels(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "returns an an array of all supported model for the translation/transcription, the larger the model, the more accurate the translation/transcription, but also the longer it takes"
    
    """
    url = f"https://media-transcriber.p.rapidapi.com/getModels"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "media-transcriber.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettranscribedlanguages(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "returns an an object containing all the supported transcribed languages code associated with their language."
    
    """
    url = f"https://media-transcriber.p.rapidapi.com/getTranscribedLanguages"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "media-transcriber.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettranslatedlanguages(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "returns an an object containing all the supported translated languages code associated with their language."
    
    """
    url = f"https://media-transcriber.p.rapidapi.com/getTranslatedLanguages"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "media-transcriber.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

