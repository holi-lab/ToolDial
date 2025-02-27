import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def entradas(diccionario: str, cantidad: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Regresa las entradas del diccionario especificado por defecto 20"
    
    """
    url = f"https://diccionariosv5.p.rapidapi.com/entradas"
    querystring = {'diccionario': diccionario, }
    if cantidad:
        querystring['cantidad'] = cantidad
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "diccionariosv5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

