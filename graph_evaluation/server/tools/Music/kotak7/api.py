import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def tangga_lagu(index: int, num: int, lang: str, country: str, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Daftar tangga lagu"
    
    """
    url = f"https://kotak7.p.rapidapi.com/toplist/{is_id}/tracks"
    querystring = {'index': index, 'num': num, 'lang': lang, 'country': country, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kotak7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def artis_info(num: int, lang: str, index: int, country: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Artis info"
    
    """
    url = f"https://kotak7.p.rapidapi.com/artist/{is_id}/albums"
    querystring = {'num': num, 'lang': lang, 'index': index, 'country': country, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kotak7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def artis(index: int, lang: str, country: str, num: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Daftar artis"
    
    """
    url = f"https://kotak7.p.rapidapi.com/tag/{is_id}/artists"
    querystring = {'index': index, 'lang': lang, 'country': country, 'num': num, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kotak7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

