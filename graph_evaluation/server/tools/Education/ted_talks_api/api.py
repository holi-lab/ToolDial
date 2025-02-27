import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getsubtitlelanguages(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all the TED talks subtitle languages"
    
    """
    url = f"https://ted-talks-api.p.rapidapi.com/subtitle_languages"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ted-talks-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettalks(is_id: int=None, audio_lang: str='en', speaker: str='yuval_noah_harari', publish_date: str=None, topic: str='politics', subtitle_lang: str='he', max_duration: int=None, to_publish_date: str=None, from_publish_date: str=None, min_duration: int=300, record_date: str=None, to_record_date: str=None, from_record_date: str='2017-01-01', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get info about specific talks based on the provided parameters"
    is_id: ID of a desired specific talk
        audio_lang: Return talks which their audio language is the provided language, the provided value should be the language slug according to the /audio_languages endpoint
        speaker: Return talks which at least on of their speakers is the provided speaker, the provided value should be the speaker slug according to the /speakers endpoint
        publish_date: Return talks which were published on TED.com only on the exact provided date
        topic: Return talks which at least on of their topics is the provided topics, the provided value should be the speaker slug according to the /topics endpoint
        subtitle_lang: Return talks which have subtitles in the provided language, the provided value should be the language slug according to the /subtitle_languages endpoint
        max_duration: Return talks which their duration in seconds is at most the provided value
        to_publish_date: Return talks which were published on TED.com only before the provided date
        from_publish_date: Return talks which were published on TED.com only after the provided date
        min_duration: Return talks which their duration in seconds is at least the provided value
        record_date: Return talks which were recorded only in the exact provided date
        to_record_date: Return talks which were recorded only before the provided date
        from_record_date: Return talks which were recorded only after the provided date
        
    """
    url = f"https://ted-talks-api.p.rapidapi.com/talks"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    if audio_lang:
        querystring['audio_lang'] = audio_lang
    if speaker:
        querystring['speaker'] = speaker
    if publish_date:
        querystring['publish_date'] = publish_date
    if topic:
        querystring['topic'] = topic
    if subtitle_lang:
        querystring['subtitle_lang'] = subtitle_lang
    if max_duration:
        querystring['max_duration'] = max_duration
    if to_publish_date:
        querystring['to_publish_date'] = to_publish_date
    if from_publish_date:
        querystring['from_publish_date'] = from_publish_date
    if min_duration:
        querystring['min_duration'] = min_duration
    if record_date:
        querystring['record_date'] = record_date
    if to_record_date:
        querystring['to_record_date'] = to_record_date
    if from_record_date:
        querystring['from_record_date'] = from_record_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ted-talks-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettopics(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all the TED talks topics"
    
    """
    url = f"https://ted-talks-api.p.rapidapi.com/topics"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ted-talks-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getspeakers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all the TED talks speakers"
    
    """
    url = f"https://ted-talks-api.p.rapidapi.com/speakers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ted-talks-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getaudiolanguages(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all the TED talks audio languages"
    
    """
    url = f"https://ted-talks-api.p.rapidapi.com/audio_languages"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ted-talks-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

