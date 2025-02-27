import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def status_check_video_transcription_and_summary(url: str='http://raullgdev.es/js.mp4', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will retrieve all the generated data from the provided URL.
		URLs are cached, so you will always be able to access the results of a provided video just using the same URL."
    
    """
    url = f"https://video-to-text-video-transcription-and-summarization.p.rapidapi.com/transcribe"
    querystring = {}
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "video-to-text-video-transcription-and-summarization.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

