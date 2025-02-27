import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_user_total_followers(to_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This enpoint is recommened for getting the user total followers from the profile.  ```to_id```  parameter is required and can be generated with the help of the ```/users/``` endpoint available under Twitch profile stats application on RapidAPI API hub"
    
    """
    url = f"https://twitch-profile-stats.p.rapidapi.com/users/follows"
    querystring = {'to_id': to_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitch-profile-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_profile_details(login: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This enpoint is recommened for getting the user ID (to_id) that will eventually be used on the get followers endpoint"
    
    """
    url = f"https://twitch-profile-stats.p.rapidapi.com/users"
    querystring = {'login': login, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitch-profile-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

