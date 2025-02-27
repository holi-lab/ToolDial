import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_a_random_joke(minrating: str='8', maxlength: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a random joke."
    minrating: The lowest rating the random joke is allowed to have on a scale between 0 and 10, 10 being the funniest.
        maxlength: The maximum number of characters in the joke.
        
    """
    url = f"https://webknox-jokes.p.rapidapi.com/jokes/random"
    querystring = {}
    if minrating:
        querystring['minRating'] = minrating
    if maxlength:
        querystring['maxLength'] = maxlength
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webknox-jokes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_random_one_liner(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a random one liner. This is usually a funny quote or an interesting observation."
    
    """
    url = f"https://webknox-jokes.p.rapidapi.com/jokes/oneLiner"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webknox-jokes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def find_jokes(keywords: str, numjokes: int, category: str='Chuck Norris', minrating: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find jokes with given search criteria."
    keywords: A comma-separated list of keywords that must occur in the joke.
        numjokes: The number of jokes, between 0 and 10.
        category: The category of the joke.
        minrating: The minimum rating of the joke between 0 and 10.
        
    """
    url = f"https://webknox-jokes.p.rapidapi.com/jokes/search"
    querystring = {'keywords': keywords, 'numJokes': numjokes, }
    if category:
        querystring['category'] = category
    if minrating:
        querystring['minRating'] = minrating
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webknox-jokes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def praise_somebody(name: str, content: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Praise a person that deserves to be praised with random phrases."
    name: The name of the person that deserves to be praised.
        content: The reason why the person deserves to be praised.
        
    """
    url = f"https://webknox-jokes.p.rapidapi.com/jokes/praise"
    querystring = {'name': name, 'content': content, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webknox-jokes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def insult_somebody(name: str, content: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Insult a person that deserves to be insulted."
    name: The name of the person that should be insulted.
        content: The reason why the person should get insulted.
        
    """
    url = f"https://webknox-jokes.p.rapidapi.com/jokes/insult"
    querystring = {'name': name, 'content': content, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webknox-jokes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

