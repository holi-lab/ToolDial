import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_horoscope_multilingual(language: str, period: str, sign: str, type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return horoscope information by sign, period, type and languages provided."
    language: en: English, 
zh-t: Chinese (transditional), 
zh-s: Chinese (simplified),
es: Spanish,
hi: Hindu,
fr: French,
ja: Japanese,
ko: Korean
        type: Depend on period, use **Get Types** method to get available types for the selected period.
        
    """
    url = f"https://horoscopes-ai.p.rapidapi.com/get_horoscope/{sign}/{period}/{type}/{language}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horoscopes-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_horoscope_english_only(sign: str, period: str, type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return horoscope information by sign, period, type and languages provided."
    type: Depend on period, use **Get Types** method to get available types for the selected period.
        
    """
    url = f"https://horoscopes-ai.p.rapidapi.com/get_horoscope_en/{sign}/{period}/{type}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horoscopes-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_signs(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return 12 horoscope sign names."
    
    """
    url = f"https://horoscopes-ai.p.rapidapi.com/get_signs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horoscopes-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_period(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return available period for Horoscope."
    
    """
    url = f"https://horoscopes-ai.p.rapidapi.com/get_periods"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horoscopes-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_types(period: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return available types for the particular period."
    
    """
    url = f"https://horoscopes-ai.p.rapidapi.com/get_types/{period}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horoscopes-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

