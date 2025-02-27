import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def municipio_por_c_digo_postal(codigo_postal: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Obtiene el municipio al que pertenece un código postal"
    
    """
    url = f"https://codigos-postales-de-espana.p.rapidapi.com/"
    querystring = {'codigo-postal': codigo_postal, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "codigos-postales-de-espana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

