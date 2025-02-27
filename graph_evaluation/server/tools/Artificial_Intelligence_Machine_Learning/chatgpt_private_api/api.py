import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def gettoken(email: str, password: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Used to automatically log in to the ChatGPT website and obtain the authrization token required for the API.
		This interface takes a long time to process, please be patient and wait for about 3-5 seconds.
		用来自动登录ChatGPT网站，获取API所需的authrization token
		这个接口所需时间较长，请耐心等待，约需3-5秒"
    
    """
    url = f"https://chatgpt-private-api1.p.rapidapi.com/account/gettoken"
    querystring = {'email': email, 'password': password, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chatgpt-private-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def conversation_list(limit: int, offset: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "获取会话列表，偏移从零开始，每次返回默认20笔，设置偏移可以分页查询"
    
    """
    url = f"https://chatgpt-private-api1.p.rapidapi.com/conversation/list"
    querystring = {'limit': limit, 'offset': offset, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chatgpt-private-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

