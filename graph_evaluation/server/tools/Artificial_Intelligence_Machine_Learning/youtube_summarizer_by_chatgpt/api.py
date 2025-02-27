import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def youtube_summarizer_v2(url: str, key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Insert a YouTube URL which has subtitles (captions) available. It will return a summary of the video (max 300 words). This API requires OpenAI API key (get here for free- https://platform.openai.com/account/api-keys)"
    key: Get ChatGPT API key (free): https://platform.openai.com/account/api-keys
        
    """
    url = f"https://youtube-summarizer-by-chatgpt.p.rapidapi.com/ytsummary1/"
    querystring = {'url': url, 'key': key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-summarizer-by-chatgpt.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

