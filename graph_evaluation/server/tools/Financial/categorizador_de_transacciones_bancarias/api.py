import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getcategoria(glosa: str='Starbucks', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Ingresar glosa de texto descriptiva de la transacción bancaria para obtener categoría y listado de palabras descriptivas para la transacción. Por ejemplo "Starbucks Caja1" -> "Comida y Bebidas""
    
    """
    url = f"https://categorizador-de-transacciones-bancarias.p.rapidapi.com/resources/categorizacion/categoria"
    querystring = {}
    if glosa:
        querystring['glosa'] = glosa
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "categorizador-de-transacciones-bancarias.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

