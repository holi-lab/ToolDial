import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def plaque(plaque: str, with_k_type: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Plaque d'immatriculation française"
    with_k_type: Demandez le code TecDoc k-Type en définissant l'attribut with_k_type=1 , par défaut est 0.
        
    """
    url = f"https://api-de-plaque-d-immatriculation-france.p.rapidapi.com/{plaque}"
    querystring = {}
    if with_k_type:
        querystring['with_k_type'] = with_k_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-de-plaque-d-immatriculation-france.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

