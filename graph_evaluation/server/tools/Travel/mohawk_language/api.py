import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def json(unique_id: str, say: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Text and Secret Commands to send to Bot. Bot responds with JSON"
    unique_id: unique conversation id.
        say: Text and Secret Commands sent to Bot.
        
    """
    url = f"https://monigarr-mohawk-language-bot-v1.p.rapidapi.com/conversation_start.php?bot_id=1&say={say}&convo_id={unique_id}&format=json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "monigarr-mohawk-language-bot-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

