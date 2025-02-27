import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def onyomi_reading(on: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Advanced Search URL parameters are described here as individual endpoints to permit per parameter testing. Parameters may be combined as required."
    on: Katakana or romaji
        
    """
    url = f"https://kanjialive-api.p.rapidapi.com/api/public/search/advanced/"
    querystring = {'on': on, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kanjialive-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def kunyomi_reading(kun: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Advanced Search URL parameters are described here as individual endpoints to permit per parameter testing. Parameters may be combined as required."
    kun: Hiragana or romaji
        
    """
    url = f"https://kanjialive-api.p.rapidapi.com/api/public/search/advanced/"
    querystring = {'kun': kun, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kanjialive-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def kanji_english_meaning(kem: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Advanced Search URL parameters are described here as individual endpoints to permit per parameter testing. Parameters may be combined as required."
    kem: Simplified English kanji meaning
        
    """
    url = f"https://kanjialive-api.p.rapidapi.com/api/public/search/advanced/"
    querystring = {'kem': kem, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kanjialive-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def kanji_stroke_number(ks: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Advanced Search URL parameters are described here as individual endpoints to permit per parameter testing. Parameters may be combined as required."
    ks: Positive integer
        
    """
    url = f"https://kanjialive-api.p.rapidapi.com/api/public/search/advanced/"
    querystring = {'ks': ks, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kanjialive-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def radical_english_meaning(rem: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Advanced Search URL parameters are described here as individual endpoints to permit per parameter testing. Parameters may be combined as required."
    rem: Radical's meaning in English
        
    """
    url = f"https://kanjialive-api.p.rapidapi.com/api/public/search/advanced/"
    querystring = {'rem': rem, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kanjialive-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def radical_position(rpos: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Advanced Search URL parameters are described here as individual endpoints to permit per parameter testing. Parameters may be combined as required."
    rpos: Hiragana or romaji
        
    """
    url = f"https://kanjialive-api.p.rapidapi.com/api/public/search/advanced/"
    querystring = {'rpos': rpos, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kanjialive-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def radical_stroke_number(rs: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Advanced Search URL parameters are described here as individual endpoints to permit per parameter testing. Parameters may be combined as required."
    rs: Positive integer
        
    """
    url = f"https://kanjialive-api.p.rapidapi.com/api/public/search/advanced/"
    querystring = {'rs': rs, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kanjialive-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def study_list_ap_exam(list: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Advanced Search URL parameters are described here as individual endpoints to permit per parameter testing. Parameters may be combined as required."
    list: Kanji divided into chapters (1-20) by Kanji alive
        
    """
    url = f"https://kanjialive-api.p.rapidapi.com/api/public/search/advanced/"
    querystring = {'list': list, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kanjialive-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def study_list_macquarie(list: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Advanced Search URL parameters are described here as individual endpoints to permit per parameter testing. Parameters may be combined as required."
    list: Kanji divided into chapters (12-22)
        
    """
    url = f"https://kanjialive-api.p.rapidapi.com/api/public/search/advanced"
    querystring = {'list': list, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kanjialive-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def kanji_grade_level(grade: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Advanced Search URL parameters are described here as individual endpoints to permit per parameter testing. Parameters may be combined as required."
    grade: Positive integer
        
    """
    url = f"https://kanjialive-api.p.rapidapi.com/api/public/search/advanced/"
    querystring = {'grade': grade, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kanjialive-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_kanji_details(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Language attributes and media files for all supported kanji (~ 6.5MB)"
    
    """
    url = f"https://kanjialive-api.p.rapidapi.com/api/public/kanji/all"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kanjialive-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

