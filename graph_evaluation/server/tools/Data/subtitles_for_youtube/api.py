import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_subtitle_in_srt_format(videoid: str, type: str=None, translated: str=None, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get subtitle in SRT format"
    videoid: YouTube video Id
        type: If you want to force either human generated subs or auto-generated ones use this param. By default if the query param is not provided then API will first try to find a human generated sub, on failing it will fall back to auto-generated sub
        translated: If subtitle is not present for a particular language, then we auto-translate it to the requested language. This behaviour can be disabled by passing translated=original, in that case if no subtitle is present in the requested language, nothing will be returned
        lang: Get subtitle in this language, if not specified default language of the video is picked. The languages codes can be retrieved using the List languages endpoint
        
    """
    url = f"https://subtitles-for-youtube.p.rapidapi.com/subtitles/{videoid}.srt"
    querystring = {}
    if type:
        querystring['type'] = type
    if translated:
        querystring['translated'] = translated
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "subtitles-for-youtube.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_subtitle_in_json_format(videoid: str, translated: str=None, lang: str=None, type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get subtitle as a JSON object"
    videoid: YouTube video Id
        translated: If subtitle is not present for a particular language, then we auto-translate it to the requested language. This behaviour can be disabled by passing translated=original, in that case if no subtitle is present in the requested language, nothing will be returned
        lang: Get subtitle in this language, if not specified default language of the video is picked. The languages codes can be retrieved using the List languages endpoint
        type: If you want to force either human generated subs or auto-generated ones use this param. By default if the query param is not provided then API will first try to find a human generated sub, on failing it will fall back to auto-generated sub
        
    """
    url = f"https://subtitles-for-youtube.p.rapidapi.com/subtitles/{videoid}"
    querystring = {}
    if translated:
        querystring['translated'] = translated
    if lang:
        querystring['lang'] = lang
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "subtitles-for-youtube.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_languages(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all supported languages"
    
    """
    url = f"https://subtitles-for-youtube.p.rapidapi.com/subtitles/languages"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "subtitles-for-youtube.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_all_available_subtitles(videoid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all available subtitles for a YouTube video"
    videoid: YouTube video Id
        
    """
    url = f"https://subtitles-for-youtube.p.rapidapi.com/subtitles/{videoid}/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "subtitles-for-youtube.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

