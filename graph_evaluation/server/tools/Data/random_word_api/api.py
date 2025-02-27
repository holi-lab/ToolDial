import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_word_by_length_and_start(length: int, start: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a random word of specified length and that starts with specified string.
		For example, 7 and "fru" will return any word that is 7 letters long and starts with "fru", such as "fruiter"."
    
    """
    url = f"https://random-word-api.p.rapidapi.com/LS/{length}/{start}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "random-word-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_word_by_contain(substring: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a random word that contains a certain string.
		For example, if the string is "le" then you will get any word that contains the string, whether it's at **the start, the end or the middle**, and you will get for example "odourless"."
    
    """
    url = f"https://random-word-api.p.rapidapi.com/C/{substring}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "random-word-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_word_by_start(start: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a random word that starts with the specified string.
		For example, "fru" will only return words that start with "fru", such as "fruit""
    
    """
    url = f"https://random-word-api.p.rapidapi.com/S/{start}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "random-word-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_word_by_length_and_contain(substring: str, length: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a random word that contains a certain string and contains a certain amount of letters.
		For example, if the substring is "le" and the length "7", you will get a word that contains "le" and is 7 characters long, such as "dozzled"."
    
    """
    url = f"https://random-word-api.p.rapidapi.com/LC/{length}/{substring}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "random-word-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_word_by_start_and_contain(substring: str, start: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a random word that starts with a certain string and contains a certain string.
		For example, if the start is "fru" and the substring is "le", you will get a word that starts with "fru" and contains "le", such as "frustules"."
    
    """
    url = f"https://random-word-api.p.rapidapi.com/SC/{start}/{substring}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "random-word-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_word_by_length_start_and_contain(start: str, length: int, substring: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a random word that is a certain amount of characters long, starts with a certain string and contains a certain string.
		For example, if the length is 7, the start "fru" and the substring "le", you will get a 7 letter long word that begins with fru and contains le, like "frumple"."
    
    """
    url = f"https://random-word-api.p.rapidapi.com/LSC/{length}/{start}/{substring}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "random-word-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_word_by_length(length: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a random word that has the specified length. For example, 7 will return any word that is 7 letters long, such as "swallow"."
    
    """
    url = f"https://random-word-api.p.rapidapi.com/L/{length}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "random-word-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_word(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a random word. No parameters. Just randomness."
    
    """
    url = f"https://random-word-api.p.rapidapi.com/get_word"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "random-word-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_french_word(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Same as random word, but grabs a random french word instead."
    
    """
    url = f"https://random-word-api.p.rapidapi.com/french_word"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "random-word-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

