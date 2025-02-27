import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def send_message(m: str, bot_token: str, chat_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this telegram bot to send your message to any group!"
    m: Your message text to send
        bot_token: Get your token in telegram with BOT Father.
        chat_id: If group, add -100 before of Channel ID
        
    """
    url = f"https://telegram-send-message.p.rapidapi.com/send"
    querystring = {'m': m, 'bot_token': bot_token, 'chat_id': chat_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "telegram-send-message.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

