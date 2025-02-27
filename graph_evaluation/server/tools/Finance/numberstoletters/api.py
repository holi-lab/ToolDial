import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def convertir_cantidad_a_letra_moneda_mxn_ingles(moneda: str, monto: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convertir cantidad a letra Moneda MXN en Ingles"
    
    """
    url = f"https://numberstoletters.p.rapidapi.com/convert-english"
    querystring = {'moneda': moneda, 'monto': monto, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "numberstoletters.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convertir_cantidad_a_letra_moneda_mxn_espa_ol(moneda: str, monto: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convierte de cantidad a letras pesos Mexicano EndPoind Español
		Se agrego el parametro **moneda**, los tipos aceptados para este parametro son los siguientes (PESOS, DOLARES, EUROS), TODO EN MAYUSCULAS."
    
    """
    url = f"https://numberstoletters.p.rapidapi.com/convert"
    querystring = {'moneda': moneda, 'monto': monto, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "numberstoletters.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

