import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def verificar_n_mero_en_repep(numero: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Verifica si un número esta registrado en el REPEP."
    
    """
    url = f"https://registro-publico-para-evitar-publicidad-repep.p.rapidapi.com/api/v2/repep/{numero}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "registro-publico-para-evitar-publicidad-repep.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

