import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def informaci_n_de_n_mero(numero: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Obtener información de un número en partiocular."
    
    """
    url = f"https://informacion-de-cualquier-numero-fijo-y-movil-de-mexico.p.rapidapi.com/api/v11/telefono/{numero}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "informacion-de-cualquier-numero-fijo-y-movil-de-mexico.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

