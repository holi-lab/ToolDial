import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def echo(authorization: str=None, msg: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    authorization: the client's access token
        msg: a message to be echoed
        
    """
    url = f"https://oauthecho.p.rapidapi.com/echo"
    querystring = {}
    if authorization:
        querystring['Authorization'] = authorization
    if msg:
        querystring['msg'] = msg
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "oauthecho.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def token(client_secret: str=None, client_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    client_secret: the client's secret
        client_id: the client's id
        
    """
    url = f"https://oauthecho.p.rapidapi.com/token"
    querystring = {}
    if client_secret:
        querystring['client_secret'] = client_secret
    if client_id:
        querystring['client_id'] = client_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "oauthecho.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

