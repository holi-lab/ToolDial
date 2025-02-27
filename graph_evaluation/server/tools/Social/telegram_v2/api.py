import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def telegram_group_info(group: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint returns realtime information about  public Telegram Group
		
		- Title
		- Photo
		- Description
		- Members
		- On line Memebers"
    group: The @username of the group

        
    """
    url = f"https://telegram7.p.rapidapi.com/group/{group}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "telegram7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def messages_from_telegram_channel(channel: str, idmessage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint returns last 10 messages  or specific id message from Telegram Channel. 
		The channel must be public.
		
		- Author
		- Photo
		- Message
		- Views 
		- DateTime
		- ID"
    channel: The @username of the channel
        idmessage: ID of a single message to view
        
    """
    url = f"https://telegram7.p.rapidapi.com/messages/{channel}/{idMessage}"
    querystring = {}
    if idmessage:
        querystring['idMessage'] = idmessage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "telegram7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def telegram_channel_info(channel: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint returns realtime information about  public Telegram Channel
		
		- Title
		- Photo
		- Description
		- Members"
    channel: The @username of the channel
        
    """
    url = f"https://telegram7.p.rapidapi.com/channel/{channel}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "telegram7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

