import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def mind_id_nuke(mind_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Delete your mind's concept network and start it again from scratch."
    mind_id: The unique string you chose to identify the mind with.
        
    """
    url = f"https://mind.p.rapidapi.com/{mind_id}/nuke"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mind.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mind_id_mind_name_user_name_message(user_name: str, mind_id: str, mind_name: str, message: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Used to send a message to a bot, or just to create a new one. Params are: a unique ID for the bot, or "mind" (can be anything, an ID always returns the same mind), the name of the mind (can be changed as often as desired), the name of the person interacting with the mind (can be changed as often as desired), and the message to send to the mind."
    user_name: Name of the user
        mind_id: A unique string to identify the mind, like an account username for the bot. (Note when choosing: "Mind", "mind", "Mind1", and "mind1" all return your default mind.)
        mind_name: Name the mind should go by
        message: Message to the mind. If you have never used a mind-id before, the first message will also trigger the creation of the mind.
        
    """
    url = f"https://mind.p.rapidapi.com/{mind_id}/{mind_name}/{user_name}/{message}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mind.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

