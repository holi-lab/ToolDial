import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def palindrome_check(string: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint takes a GET request with a query parameter "string" and returns a JSON object that includes the result whether the string is palindrome or not."
    
    """
    url = f"https://word-wizard-api.p.rapidapi.com/palindrome"
    querystring = {'string': string, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "word-wizard-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def anagram_check(word1: str, word2: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint takes two GET request with query parameter "word1" and "word2" and returns a JSON object that includes the result whether the two words are anagram or not."
    
    """
    url = f"https://word-wizard-api.p.rapidapi.com/anagram"
    querystring = {'word1': word1, 'word2': word2, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "word-wizard-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def word_stats(string: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint takes a GET request with a query parameter "string" and returns a JSON object that includes the count of the each unique word in the string"
    
    """
    url = f"https://word-wizard-api.p.rapidapi.com/wordstats"
    querystring = {'string': string, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "word-wizard-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def word_count(string: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint takes a GET request with a query parameter "string" and returns the number of words in the string."
    
    """
    url = f"https://word-wizard-api.p.rapidapi.com/wordcount"
    querystring = {'string': string, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "word-wizard-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def capitalize_string(string: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint takes a GET request with a query parameter "string" and returns the capitalized string"
    
    """
    url = f"https://word-wizard-api.p.rapidapi.com/capitalize"
    querystring = {'string': string, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "word-wizard-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reverse_string(string: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint takes a GET request with a query parameter "string" and returns the reverse of that string"
    
    """
    url = f"https://word-wizard-api.p.rapidapi.com/reverse"
    querystring = {'string': string, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "word-wizard-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

