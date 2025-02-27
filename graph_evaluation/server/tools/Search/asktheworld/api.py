import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getcountrycode(getcountrycodes: str='true', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets all Country Codes which you can use to search for a specific country."
    
    """
    url = f"https://asktheworld.p.rapidapi.com/"
    querystring = {}
    if getcountrycodes:
        querystring['getCountryCodes'] = getcountrycodes
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "asktheworld.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getquestions(keyword: str, countrycode: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets Questions Asked by Keyword from Search Engines"
    
    """
    url = f"https://asktheworld.p.rapidapi.com/"
    querystring = {'keyword': keyword, }
    if countrycode:
        querystring['countryCode'] = countrycode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "asktheworld.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmultiplequestions(keyword: str, countrycode: str, multiplequestions: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets casual questions asked on Search Engine with your keyword."
    
    """
    url = f"https://asktheworld.p.rapidapi.com/"
    querystring = {'keyword': keyword, 'countryCode': countrycode, }
    if multiplequestions:
        querystring['multipleQuestions'] = multiplequestions
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "asktheworld.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

