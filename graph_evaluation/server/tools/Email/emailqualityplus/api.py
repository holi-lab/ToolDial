import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def valida_o_de_email(email: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Validação Simples de Email"
    email: email a ser validado
        
    """
    url = f"https://emailqualityplus.p.rapidapi.comhttps://api.datamotion.com.br/emailqualityplus/v1/api/EmailValidation/MaShapeRapidAPI"
    querystring = {'email': email, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emailqualityplus.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

