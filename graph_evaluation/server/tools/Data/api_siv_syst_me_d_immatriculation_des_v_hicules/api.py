import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getcarinfo(plaque: str, with_k_type: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides car info from a French car plate, all you need is to interrogate the path with the correct formatted car plate, example "/AA206bc". And you will get all info you need"
    with_k_type: Demandez le code TecDoc k-Type en définissant l'attribut with_k_type=1 , par défaut est 0.

        
    """
    url = f"https://api-siv-systeme-d-immatriculation-des-vehicules.p.rapidapi.com/{plaque}"
    querystring = {}
    if with_k_type:
        querystring['with_k_type'] = with_k_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-siv-systeme-d-immatriculation-des-vehicules.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

