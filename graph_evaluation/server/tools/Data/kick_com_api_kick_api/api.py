import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_channel_clips(cursor: str, channel_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Channel Clips"
    
    """
    url = f"https://kick-com-api-kick-api.p.rapidapi.com/get_channel_clips"
    querystring = {'cursor': cursor, 'channel_name': channel_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kick-com-api-kick-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_chat_user_like_user_from_chat(channel_name: str, username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Chat User (like user from chat)"
    
    """
    url = f"https://kick-com-api-kick-api.p.rapidapi.com/get_chat_user"
    querystring = {'channel_name': channel_name, 'username': username, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kick-com-api-kick-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_chatroom_rules(channel_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Chatroom Rules"
    
    """
    url = f"https://kick-com-api-kick-api.p.rapidapi.com/get_chatroom_rules"
    querystring = {'channel_name': channel_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kick-com-api-kick-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_channel_links(channel_name: str='gmhikaru', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Channel Links"
    
    """
    url = f"https://kick-com-api-kick-api.p.rapidapi.com/get_channel_links"
    querystring = {}
    if channel_name:
        querystring['channel_name'] = channel_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kick-com-api-kick-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_channel_polls(channel_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Channel Polls"
    
    """
    url = f"https://kick-com-api-kick-api.p.rapidapi.com/get_channel_polls"
    querystring = {'channel_name': channel_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kick-com-api-kick-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_channel_chatroom(channel_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Channel Chatroom"
    
    """
    url = f"https://kick-com-api-kick-api.p.rapidapi.com/get_channel_chatroom"
    querystring = {'channel_name': channel_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kick-com-api-kick-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_channel_emotes(channel_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Channel Emotes"
    
    """
    url = f"https://kick-com-api-kick-api.p.rapidapi.com/get_channel_emotes"
    querystring = {'channel_name': channel_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kick-com-api-kick-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_channel_livestream_info(channel_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Channel Livestream Info"
    
    """
    url = f"https://kick-com-api-kick-api.p.rapidapi.com/get_channel_livestream_info"
    querystring = {'channel_name': channel_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kick-com-api-kick-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_channel_details(channel_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Channel Details"
    
    """
    url = f"https://kick-com-api-kick-api.p.rapidapi.com/get_channel_details"
    querystring = {'channel_name': channel_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kick-com-api-kick-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_chat_messages(livestream_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns chat messages from livestream ID."
    
    """
    url = f"https://kick-com-api-kick-api.p.rapidapi.com/get_chat_messages"
    querystring = {'livestream_id': livestream_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kick-com-api-kick-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_livestreams_from_category(page: str, category: str, limit: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns livestreams from category."
    
    """
    url = f"https://kick-com-api-kick-api.p.rapidapi.com/get_livestreams"
    querystring = {'page': page, 'category': category, 'limit': limit, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kick-com-api-kick-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_subcategories(page: str, limit: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns subcategories."
    
    """
    url = f"https://kick-com-api-kick-api.p.rapidapi.com/get_subcategories"
    querystring = {'page': page, 'limit': limit, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kick-com-api-kick-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

