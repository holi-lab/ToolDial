import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def send_image(send_image: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows users to send images"
    
    """
    url = f"https://ai-messaging.p.rapidapi.com/send-image"
    querystring = {}
    if send_image:
        querystring['send-image'] = send_image
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ai-messaging.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def receive_text(receive_text: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows users to receive text messages."
    receive_text: This endpoint allows users to receive text messages.
        
    """
    url = f"https://ai-messaging.p.rapidapi.com/receive-text"
    querystring = {}
    if receive_text:
        querystring['receive-text'] = receive_text
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ai-messaging.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def send_text(send_text: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will allow users to send text messages."
    send_text: This endpoint will allow users to send text messages
        
    """
    url = f"https://ai-messaging.p.rapidapi.com/send-text"
    querystring = {}
    if send_text:
        querystring['send-text'] = send_text
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ai-messaging.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

