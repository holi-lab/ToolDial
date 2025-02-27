import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def list_language_options(accept: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the languages available (used in other endpoints)"
    accept: Accept Header
        
    """
    url = f"https://ocr-text-extractor.p.rapidapi.com/languages/list-options"
    querystring = {'Accept': accept, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ocr-text-extractor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_language_options(accept: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the languages available (used in other endpoints)"
    accept: Accept Header
        
    """
    url = f"https://ocr-text-extractor.p.rapidapi.com/languages/list-options"
    querystring = {'Accept': accept, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ocr-text-extractor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_ocr_engine_options(accept: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the optical character engines available (used in other endpoints)"
    accept: Accept Header
        
    """
    url = f"https://ocr-text-extractor.p.rapidapi.com/ocr-engines/list-options"
    querystring = {'Accept': accept, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ocr-text-extractor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

