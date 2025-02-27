import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def dictionary_synonyms_temporarily_unavailable(language: str, q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Dictionary Synonyms"
    
    """
    url = f"https://webit-language.p.rapidapi.com/dictionary-synonyms"
    querystring = {'language': language, 'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webit-language.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dictionary_lookup_temporarily_unavailable(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Dictionary Lookup"
    
    """
    url = f"https://webit-language.p.rapidapi.com/dictionary-lookup"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webit-language.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dictionary_examples_temporarily_unavailable(q: str, language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Dictionary Examples"
    
    """
    url = f"https://webit-language.p.rapidapi.com/dictionary-examples"
    querystring = {'q': q, 'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webit-language.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dictionary_antonyms_temporarily_unavailable(language: str, q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Dictionary Antonyms"
    
    """
    url = f"https://webit-language.p.rapidapi.com/dictionary-antonyms"
    querystring = {'language': language, 'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webit-language.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bilingual_dictionary(q: str, is_from: str, to: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Powerful multi-language bilingual neural translation dictionary, supporting 80+ languages."
    q: The word or comma separated words (up to 10 words per request) to seek bilingual translations for.
        from: Supported  ISO 639-1 language codes: 'ar', 'az', 'be', 'bg', 'ca', 'ce', 'cs', 'cy', 'da', 'de', 'el', 'en', 'eo', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'gl', 'he', 'hi', 'hr', 'hu', 'hy', 'id', 'it', 'ja', 'ka', 'kk', 'ko', 'la', 'lt', 'lv', 'mk', 'ms', 'nl', 'nn', 'no', 'pl', 'pt', 'ro', 'ru', 'sh', 'sk', 'sl', 'sr', 'sv', 'ta', 'tg', 'th', 'tr', 'tt', 'uk', 'ur', 'uz', 'vi', 'vo', 'zh'.
        to: Supported  ISO 639-1 language codes: 'ar', 'az', 'be', 'bg', 'ca', 'ce', 'cs', 'cy', 'da', 'de', 'el', 'en', 'eo', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'gl', 'he', 'hi', 'hr', 'hu', 'hy', 'id', 'it', 'ja', 'ka', 'kk', 'ko', 'la', 'lt', 'lv', 'mk', 'ms', 'nl', 'nn', 'no', 'pl', 'pt', 'ro', 'ru', 'sh', 'sk', 'sl', 'sr', 'sv', 'ta', 'tg', 'th', 'tr', 'tt', 'uk', 'ur', 'uz', 'vi', 'vo', 'zh'.
        
    """
    url = f"https://webit-language.p.rapidapi.com/dictionary-bilingual"
    querystring = {'q': q, 'from': is_from, 'to': to, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webit-language.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def transliterate_any_to_latin(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Transliteration endpoint provides you with transliteration of text from any language to Latin. Transliterate "こんにちは" to "kon'nichiha" with ease."
    text: Transliterate any text to Latin.
        
    """
    url = f"https://webit-language.p.rapidapi.com/transliterate"
    querystring = {'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webit-language.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def autocomplete(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides autocomplete (word completion) for 80+ languages."
    
    """
    url = f"https://webit-language.p.rapidapi.com/autocomplete"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webit-language.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

