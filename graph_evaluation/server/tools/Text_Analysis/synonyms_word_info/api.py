import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_v1_word_synonyms(str: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns synonyms and related words about a morpheme (any word, short phrase, or abbreviation). DOCUMENTATION: https://wordio.co/api
		
		type:
		best_by_pos, all_by_pos, all_by_relevance,all_by_relevance"
    str: Word or short phrase like \"doctor\" or \"medical doctor\". Short or long: \"i\" or \"water under the bridge\". Small words, especially single letter are experimental.
        
    """
    url = f"https://synonyms-word-info.p.rapidapi.com/v1/word/synonyms"
    querystring = {'str': str, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "synonyms-word-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_v1_word_info(str: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns information about a word - is it plural, singular, conjunction? - its root form, abbreviations, acronyms - parts of speech - sentiment analysis score (positive/neutral or negative). DOCUMENTATION at https://wordio.co/api"
    str: Word or short phrase like \"doctor\" or \"medical doctor\". Short or long: \"i\" or \"water under the bridge\". Small words, especially single letter are experimental.
        
    """
    url = f"https://synonyms-word-info.p.rapidapi.com/v1/word/info"
    querystring = {}
    if str:
        querystring['str'] = str
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "synonyms-word-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_v1_word(str: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns synonyms grouped by part-of-speech and by sentiment. Returns information about a word - is it plural, singular, conjunction? - its root form, abbreviations, acronyms - parts of speech - sentiment analysis score (positive/neutral or negative). DOCUMENTATION at https://wordio.co/api"
    str: Word or short phrase like "doctor" or "medical doctor". Short or long: "i" or "water under the bridge". Small words, especially single letter are experimental.
        
    """
    url = f"https://synonyms-word-info.p.rapidapi.com/v1/word"
    querystring = {'str': str, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "synonyms-word-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

