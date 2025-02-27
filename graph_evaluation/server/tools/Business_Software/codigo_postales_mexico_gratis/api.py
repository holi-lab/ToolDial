import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def codigo_postal_mexicano(cp: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Escribe el numero de codigo postal mexicano"
    
    """
    url = f"https://codigo-postales-mexico-gratis.p.rapidapi.com/code_postal/consulta/cp.php"
    querystring = {'cp': cp, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "codigo-postales-mexico-gratis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

