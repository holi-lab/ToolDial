import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def transcript_by_youtube_video_id(youtubeid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "enter youtube video id, for example:https://www.youtube.com/watch?v=rDiGYuQicpA (rDiGYuQicpA is the id)"
    youtubeid: youtube id of the desired transcript
        
    """
    url = f"https://bestapi-ted-v1.p.rapidapi.com/transcriptById"
    querystring = {'youtubeId': youtubeid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bestapi-ted-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def talk_by_youtube_id(youtubeid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get talk details by its youTube Id"
    youtubeid: talk youTube Id - for example from url: https://www.youtube.com/watch?v=rDiGYuQicpA
        
    """
    url = f"https://bestapi-ted-v1.p.rapidapi.com/talksByYoutubeId"
    querystring = {'youtubeId': youtubeid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bestapi-ted-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def talks_by_speaker(speaker: str, size: int=3, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "enter speaker name and get all talks by this speaker"
    speaker: speaker name (not case sensitive)
        size: amount of results to return
        
    """
    url = f"https://bestapi-ted-v1.p.rapidapi.com/bySpeaker"
    querystring = {'speaker': speaker, }
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bestapi-ted-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def talks_by_name(name: str, size: int=3, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "enter talk name and get talk details"
    name: get talks details by name (free text search)
        size: number of results
        
    """
    url = f"https://bestapi-ted-v1.p.rapidapi.com/talksByName"
    querystring = {'name': name, }
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bestapi-ted-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def transcripts_text_search(text: str, size: int=3, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "we will look in all ted talks transcripts for your input text (free text search)"
    text: enter text to search in all talks transcript
        size: amount of results to return
        
    """
    url = f"https://bestapi-ted-v1.p.rapidapi.com/transcriptFreeText"
    querystring = {'text': text, }
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bestapi-ted-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def talks_by_description(description: str, size: int=3, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search ted talks by talks description"
    description: search ted talks by talk description (free text search)
        size: amount of results
        
    """
    url = f"https://bestapi-ted-v1.p.rapidapi.com/talksByDescription"
    querystring = {'description': description, }
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bestapi-ted-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

