import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_1_obter_c_digo_do_chat_code_to_chat(co_uasg: int, numprp: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Obter Código do Chat"
    co_uasg: Código UASG do Orgão
        numprp: Número do Processo a ser Consultado
        
    """
    url = f"https://compras-net-api.p.rapidapi.com/codigo"
    querystring = {'co_uasg': co_uasg, 'numprp': numprp, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "compras-net-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_2_mensagens_do_chat_chat_message(cod: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Chat"
    cod: Código do Chat (Pode ser obtido no outro endpoint)
        
    """
    url = f"https://compras-net-api.p.rapidapi.com/chat"
    querystring = {'cod': cod, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "compras-net-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

