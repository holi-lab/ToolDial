import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_sessions(character_id: str, page_size: int=25, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of sessions and their corresponding `session_id`."
    
    """
    url = f"https://optitalk.p.rapidapi.com/api/chat/sessions"
    querystring = {'character_id': character_id, }
    if page_size:
        querystring['page_size'] = page_size
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "optitalk.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_session_messages(character_id: str, session_id: str='e5f6g7h8', page: str='1', page_size: str='40', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the message history inside a session."
    
    """
    url = f"https://optitalk.p.rapidapi.com/api/chat"
    querystring = {'character_id': character_id, }
    if session_id:
        querystring['session_id'] = session_id
    if page:
        querystring['page'] = page
    if page_size:
        querystring['page_size'] = page_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "optitalk.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_characters(page_size: int=25, page: int=1, private: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of publicly available characters as well as the characters that you created.
		
		## Character Usage
		Once you've retrieved a character, you can copy its ID and use it in the /chat endpoint in order to chat with it."
    private: Whether to only return characters that you've created or return all publicly available characters.
        
    """
    url = f"https://optitalk.p.rapidapi.com/api/characters"
    querystring = {}
    if page_size:
        querystring['page_size'] = page_size
    if page:
        querystring['page'] = page
    if private:
        querystring['private'] = private
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "optitalk.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

