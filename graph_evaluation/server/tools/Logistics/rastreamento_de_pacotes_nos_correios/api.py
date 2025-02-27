import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def detalhe_de_um_pacote(package: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Esse endpoint retorna todos os detalhes de um pacote específico."
    
    """
    url = f"https://rastreamento-de-pacotes-nos-correios.p.rapidapi.com/packages/{package}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rastreamento-de-pacotes-nos-correios.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

