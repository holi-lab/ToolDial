import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ip_history(ip: str, days: int=2, lang: str='en', contents: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "{Beta version}
		You can get IP torrent using history by HTTP GET request:
		
		https://{host}/history/peer?ip={ip}&days={days}&contents={contents}&lang={lang}&key={key}
		
		where
		host - hostname, will be provided.
		ip - ip address which history you want to receive.
		days - optional, search history max days ago. Default value is 14. Max value is 30.
		contents - optional, max contents in response. Default value is 20. Max value is 100.
		lang - optional, language of response. Default value is "en". Supported values "en" and "ru".
		key - API Key"
    
    """
    url = f"https://torrent-data-peers.p.rapidapi.com/history/peer?"
    querystring = {'ip': ip, }
    if days:
        querystring['days'] = days
    if lang:
        querystring['lang'] = lang
    if contents:
        querystring['contents'] = contents
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "torrent-data-peers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

