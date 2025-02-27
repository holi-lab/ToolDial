import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_available_languages(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of available languages."
    
    """
    url = f"https://hablaa-dictionary-translation-hablaa-v1.p.rapidapi.com/languages/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hablaa-dictionary-translation-hablaa-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_translation(text_to_translate: str, lang_code_src_lang_code_dst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a translation by defining source and destination languages as well as a word to translate."
    
    """
    url = f"https://hablaa-dictionary-translation-hablaa-v1.p.rapidapi.com/translation/{text_to_translate}/{lang_code_src_lang_code_dst}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hablaa-dictionary-translation-hablaa-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def example_sentences(text_to_translate: str, lang_code_src_lang_code_dst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request multiple example sentences containing the word to translate."
    
    """
    url = f"https://hablaa-dictionary-translation-hablaa-v1.p.rapidapi.com/translations-examples/{text_to_translate}/{lang_code_src_lang_code_dst}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hablaa-dictionary-translation-hablaa-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def similar_words(text_to_translate: str, lang_code_src_lang_code_dst: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get translations of similar words."
    
    """
    url = f"https://hablaa-dictionary-translation-hablaa-v1.p.rapidapi.com/translations-similar/{text_to_translate}/{lang_code_src_lang_code_dst}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hablaa-dictionary-translation-hablaa-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

