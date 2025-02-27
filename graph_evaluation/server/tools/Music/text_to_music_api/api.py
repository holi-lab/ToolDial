import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def random_string2(ascii_text: str='eeefefeaaagcaaggfffddCC', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "it puts the text in the url and gets the file to listen to music."
    
    """
    url = f"https://text-to-music-api.p.rapidapi.com/play"
    querystring = {}
    if ascii_text:
        querystring['ascii_text'] = ascii_text
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "text-to-music-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_string1(ascii_text: str='IaMaTextThatWillGiveMusic', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "it puts the text in the url and gets the file to listen to music."
    
    """
    url = f"https://text-to-music-api.p.rapidapi.com/play"
    querystring = {}
    if ascii_text:
        querystring['ascii_text'] = ascii_text
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "text-to-music-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

