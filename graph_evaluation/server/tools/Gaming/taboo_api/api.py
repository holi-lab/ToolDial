import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_result_from_category(category: str, forbiddenwordlimit: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given a stored category, get a random result which will contain the main phrase and the taboo words.
		
		Optionally pass a parameter which will control the amount of taboo words to be returned, "forbiddenWordLimit" defaults to 4 and has a max of 10"
    
    """
    url = f"https://taboo-api.p.rapidapi.com/taboo/category/{category}"
    querystring = {}
    if forbiddenwordlimit:
        querystring['forbiddenWordLimit'] = forbiddenwordlimit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taboo-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_taboo_words_from_word(word: str, forbiddenwordlimit: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide your own word and get the taboo words back.
		
		Optionally pass a parameter which will control the amount of taboo words to be returned, "forbiddenWordLimit" defaults to 4 and has a max of 10"
    
    """
    url = f"https://taboo-api.p.rapidapi.com/taboo/word/{word}"
    querystring = {}
    if forbiddenwordlimit:
        querystring['forbiddenWordLimit'] = forbiddenwordlimit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taboo-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_stored_word_from_any_category(forbiddenwordlimit: int=6, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a random result which will contain the main phrase and the taboo words, result may come from any of the stored categories.
		
		Optionally pass a parameter which will control the amount of taboo words to be returned, "forbiddenWordLimit" defaults to 4 and has a max of 10"
    
    """
    url = f"https://taboo-api.p.rapidapi.com/taboo/word"
    querystring = {}
    if forbiddenwordlimit:
        querystring['forbiddenWordLimit'] = forbiddenwordlimit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taboo-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_stored_categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of all stored categories. Use them to get results for the specified category"
    
    """
    url = f"https://taboo-api.p.rapidapi.com/taboo/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taboo-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

