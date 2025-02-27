import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def alphametics_shorter_than(size: int, number_of_results: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a JSON set of alphametics where the puzzle length is shorter than the specified size."
    size: Including +'s and =
        number_of_results: Up to your monthly limit (default is 1)
        
    """
    url = f"https://ceneezer-cryptarithm-v1.p.rapidapi.com/?smaller={size}&desc={number_of_results}"
    querystring = {}
    if number_of_results:
        querystring['number_of_results'] = number_of_results
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ceneezer-cryptarithm-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def alphametics_longer_than(size: int, number_of_results: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a JSON set of alphametics where the puzzle length is longer than the specified size."
    size: Including +'s and =
        number_of_results: Up to your monthly limit (default is 1)
        
    """
    url = f"https://ceneezer-cryptarithm-v1.p.rapidapi.com/?bigger={size}&asc={number_of_results}"
    querystring = {}
    if number_of_results:
        querystring['number_of_results'] = number_of_results
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ceneezer-cryptarithm-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def alphametics_by_letters(letters: str, number_of_results: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a JSON set of alphametics which use only the given letters"
    letters: (Up to 10, all together) the letters the alphametic is created with
        number_of_results: Up to your monthly limit (default is 1)
        
    """
    url = f"https://ceneezer-cryptarithm-v1.p.rapidapi.com/?letters={letters}&random={number_of_results}"
    querystring = {}
    if number_of_results:
        querystring['number_of_results'] = number_of_results
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ceneezer-cryptarithm-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def alphametics_by_unique_id(is_id: int, number_of_results: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a JSON set of alphametics before the supplied ID"
    id: Unsigned BigInt >0
        number_of_results: Up to your monthly limit (default is 1)
        
    """
    url = f"https://ceneezer-cryptarithm-v1.p.rapidapi.com/?id={is_id}&desc={number_of_results}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ceneezer-cryptarithm-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_alphametics(number_of_results: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return a JSON set of random alphametics"
    number_of_results: Up to your monthly limit (default is 1)
        
    """
    url = f"https://ceneezer-cryptarithm-v1.p.rapidapi.com/?random={number_of_results}"
    querystring = {}
    if number_of_results:
        querystring['number_of_results'] = number_of_results
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ceneezer-cryptarithm-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def alphametics_by_complexity(complexity: int, number_of_results: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a JSON set of alphametics that use only {int:complexity} letters"
    complexity: The number of letters in the desired alphametic
        number_of_results: Up to your monthly limit (default is 1)
        
    """
    url = f"https://ceneezer-cryptarithm-v1.p.rapidapi.com/?complexity={complexity}&random={number_of_results}"
    querystring = {}
    if number_of_results:
        querystring['number_of_results'] = number_of_results
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ceneezer-cryptarithm-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

