import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def buscar_fipe_pela_placa(placa: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Busca dados  e preço atual do veiculo pela placa na FIPE"
    
    """
    url = f"https://consulta-fipe-pela-placa.p.rapidapi.com/consulta/{placa}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "consulta-fipe-pela-placa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hist_rico_da_fipe(placa: str, months: int=12, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Busca o histórico de preço da FIPE pela dos últimos meses. Tem um tempo de requisição um pouco maior."
    
    """
    url = f"https://consulta-fipe-pela-placa.p.rapidapi.com/fipe/historico/{placa}"
    querystring = {}
    if months:
        querystring['months'] = months
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "consulta-fipe-pela-placa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

