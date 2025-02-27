import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def usuarios(whatsapp: str, password: str, email: str, name: str='João', cpf: str='12345678900', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "banco para usuarios"
    email: Campo destinado para armazenar o email
        cpf: Campo reservado para armazenar o CPF
        
    """
    url = f"https://hotspot-social-1.p.rapidapi.com/users"
    querystring = {'whatsapp': whatsapp, 'password': password, 'email': email, }
    if name:
        querystring['name'] = name
    if cpf:
        querystring['cpf'] = cpf
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hotspot-social-1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

