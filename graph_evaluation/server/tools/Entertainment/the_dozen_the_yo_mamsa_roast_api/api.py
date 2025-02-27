import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_all_jokes_in_a_single_category(category: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return all joke  in a catagory."
    
    """
    url = f"https://the-dozen-the-yo-mamsa-roast-api.p.rapidapi.com/insults/{category}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-dozen-the-yo-mamsa-roast-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_random_categorized_and_or_filter_joke(explicit: str=None, metadata: str=None, catagory: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Allows for either randomize, categorized or filtered insults to be returned with or without metadata."
    explicit: Classification of the Joke to determine if it's content is considered explicit or otherwise NSFW.
        metadata: To get the entire Joke, including metadata <sub>(date added, contributor, catagory, content, etc.)</sub>, set this to true. Default is false.
        catagory: Jokes are catagorized based on content. To get a full list of categories, see the documentation for the `insult/categories` endpoint
        
    """
    url = f"https://the-dozen-the-yo-mamsa-roast-api.p.rapidapi.com/insult"
    querystring = {}
    if explicit:
        querystring['explicit'] = explicit
    if metadata:
        querystring['metadata'] = metadata
    if catagory:
        querystring['catagory'] = catagory
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-dozen-the-yo-mamsa-roast-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_specific_insult(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Allows for a specific insults to be returned with or without metadata."
    id: This is the UUID for a Specific Joke
        
    """
    url = f"https://the-dozen-the-yo-mamsa-roast-api.p.rapidapi.com/insult/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-dozen-the-yo-mamsa-roast-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_joke_categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of all joke categories, whose as values can be used as a path or query parameter in other endpoints."
    
    """
    url = f"https://the-dozen-the-yo-mamsa-roast-api.p.rapidapi.com/insult/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-dozen-the-yo-mamsa-roast-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

