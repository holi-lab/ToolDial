import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def twitter_lookup(screen_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a social footprint of an entity on Twitter. Covering influence metrics, user information, close network, skills, followers demographics (locations, skills, influence levels, age groups)."
    screen_name: the twitter screen name of the requested entity
        
    """
    url = f"https://navigdor-influencers-lookup.p.rapidapi.com/twitter_lookup.php"
    querystring = {'screen_name': screen_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "navigdor-influencers-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def influencers_search(skill: str='Javascript', typecastid: int=3, location: str='London', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find social media influencers in a specific interest space"
    skill: Specify the specific interest space
        typecastid: Filter for Celebs / PowerUsers / Casual / Novice users
        location: A country, state or city
        
    """
    url = f"https://navigdor-influencers-lookup.p.rapidapi.com/influencers_lookup.php"
    querystring = {}
    if skill:
        querystring['skill'] = skill
    if typecastid:
        querystring['typecastId'] = typecastid
    if location:
        querystring['location'] = location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "navigdor-influencers-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def twitter_demographics_lookup(twitter_screen_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Location, Gender and Age of twitter screen name"
    twitter_screen_name: the twitter screen name
        
    """
    url = f"https://navigdor-influencers-lookup.p.rapidapi.com/twitter_lookup_demographics.php"
    querystring = {'twitter_screen_name': twitter_screen_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "navigdor-influencers-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

