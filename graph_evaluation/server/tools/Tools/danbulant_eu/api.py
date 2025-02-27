import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def bakalari(domain: str, password: str, name: str, info: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Remove the hassle with setting up XML parser and formatting tools."
    domain: Domain of school (domain.tld)
        password: Password for the username
        name: Username
        info: Which resource on Bakalari to get
        
    """
    url = f"https://danbulant-eu.p.rapidapi.com/bakalari/{info}/{name}/{domain}/{password}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "danbulant-eu.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ester(msg: str, name: str, key: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Ester AI API"
    msg: Actual message to get answer to
        name: The name of the user (used when reffering to User)
        key: API key for Ester API. Used for differentiating users.
        id: User id
        
    """
    url = f"https://danbulant-eu.p.rapidapi.com/ester"
    querystring = {'msg': msg, 'name': name, 'key': key, 'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "danbulant-eu.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def http_status_codes(code: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets information about status codes"
    code: HTTP code to get info about
        
    """
    url = f"https://danbulant-eu.p.rapidapi.com/http/{code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "danbulant-eu.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reddit_shortener(subreddit: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets a random image from given subreddit. Might not work on sureddits that allow non-image posts."
    subreddit: Subreddit name (can include r/, but works without it too)
        
    """
    url = f"https://danbulant-eu.p.rapidapi.com/reddit/{subreddit}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "danbulant-eu.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def image_datasets(dataset: str, type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Random image from given dataset"
    dataset: Gets image from given dataset
        type: Type of image in dataset
        
    """
    url = f"https://danbulant-eu.p.rapidapi.com/datasets/{dataset}/{type}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "danbulant-eu.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_dataset_types(dataset: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of types in given dataset"
    dataset: Dataset to get list from
        
    """
    url = f"https://danbulant-eu.p.rapidapi.com/datasets/{dataset}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "danbulant-eu.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

