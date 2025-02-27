import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def dialog_stream_bing_dialog_chat_stream_get(prompt: str, token: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Эндпоинт для запроса к api Bing, передавать токен и промт"
    
    """
    url = f"https://bing-ai-api.p.rapidapi.com/bing/dialog_chat_stream"
    querystring = {'prompt': prompt, 'token': token, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bing-ai-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dialog_bing_dialog_chat_get(token: str, prompt: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Эндпоинт для запроса к api Bing, передавать токен и промт"
    
    """
    url = f"https://bing-ai-api.p.rapidapi.com/bing/dialog_chat"
    querystring = {'token': token, 'prompt': prompt, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bing-ai-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def home_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Домашняя страница, uptime push)"
    
    """
    url = f"https://bing-ai-api.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bing-ai-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stream_chat_bing_stream_chat_get(token: str, prompt: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Эндпоинт для запроса к api Bing, передавать токен и промт.
		Возвращает сообщение по мере его написания самим Bing.
		Костыльное решение"
    
    """
    url = f"https://bing-ai-api.p.rapidapi.com/bing/stream_chat"
    querystring = {'token': token, 'prompt': prompt, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bing-ai-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_token_token_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "токен обязателен в любом методе"
    
    """
    url = f"https://bing-ai-api.p.rapidapi.com/token"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bing-ai-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def chat_bing_chat_get(prompt: str, token: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Эндпоинт для запроса к api Bing, передавать токен и промт"
    
    """
    url = f"https://bing-ai-api.p.rapidapi.com/bing/chat"
    querystring = {'prompt': prompt, 'token': token, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bing-ai-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

