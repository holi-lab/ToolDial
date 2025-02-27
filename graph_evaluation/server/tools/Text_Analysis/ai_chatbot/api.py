import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def chat(uid: str, message: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The chatbot endpoint."
    uid: User Identification for personalised response and response continuity.
        message: Message that the chatbot has to respond to.
        
    """
    url = f"https://ai-chatbot.p.rapidapi.com/chat/free"
    querystring = {'uid': uid, 'message': message, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ai-chatbot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

