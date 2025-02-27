import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_latest_messages(channel: str, limit: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint fetches messages from Telegram Channel. The channel should be public.
		[How to find the telegram channel name?](https://rapidapi.com/akrakoro/api/telegram-channel/details)
		- id
		- author
		- date
		- text
		- html
		- forwarded
		- button
		- link
		- photo
		- video
		- audio
		- sticker
		- attachment
		- media_poll
		- `user_read_status`  (read/unread) (mark if this message is first been read by RapidAPI's username)"
    channel: public channel name
        limit: last message number
        
    """
    url = f"https://telegram-channel.p.rapidapi.com/channel/message"
    querystring = {'channel': channel, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "telegram-channel.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_channel_info(channel: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint returns real-time information about the public Telegram Channel.
		[How to find the telegram channel name?](https://rapidapi.com/akrakoro/api/telegram-channel/details)
		- title
		- image
		- description
		- subscribers
		- subscribers_online
		- verified"
    channel: public channel name
        
    """
    url = f"https://telegram-channel.p.rapidapi.com/channel/info"
    querystring = {'channel': channel, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "telegram-channel.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

