import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def acceso_al_chat(canal: str='canal=irc-hispano', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Conecta al usuario con el chat"
    canal: Default channel
        
    """
    url = f"https://quierochat.p.rapidapi.comhttps://www.quierochat.com/terra-chat/"
    querystring = {}
    if canal:
        querystring['canal'] = canal
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quierochat.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

