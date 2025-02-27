import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_500_words(input: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Write a YouTube Script for 500 words on any topic"
    
    """
    url = f"https://youtube-script-writer-chat-gpt-3.p.rapidapi.com/write500words"
    querystring = {'input': input, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-script-writer-chat-gpt-3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generate_1000_words(input: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Write a YouTube Script for 1000 words on any topic"
    
    """
    url = f"https://youtube-script-writer-chat-gpt-3.p.rapidapi.com/write1000words"
    querystring = {'input': input, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-script-writer-chat-gpt-3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generate_2000_words(input: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Write a YouTube Script for 2000 words on any topic"
    
    """
    url = f"https://youtube-script-writer-chat-gpt-3.p.rapidapi.com/write2000words"
    querystring = {'input': input, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-script-writer-chat-gpt-3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

