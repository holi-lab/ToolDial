import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def retorna_os_registros_dos_ltimos_dias(format: str, numero_dias: str, moeda: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " Retorna os registros da ultima ocorrência dos últimos dias"
    format: Formato de resposta, json, jsonp ou xml
        numero_dias: Numero de dias para retornar (Default: 30)
        moeda: USD-BRL, EUR-BRL...
        
    """
    url = f"https://awesomeapi-exchange.p.rapidapi.com/{format}/list/{moeda}/{numero_dias}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "awesomeapi-exchange.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def todas_as_moedas(format: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retorna a ultima ocorrência de cada moeda"
    format: Formato de resposta, json, jsonp ou xml
        
    """
    url = f"https://awesomeapi-exchange.p.rapidapi.com/{format}/all"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "awesomeapi-exchange.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retorna_os_registros_de_um_per_odo_espec_fico(moeda: str, format: str, end_date: str='20190228', start_date: str='20190201', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " Retorna os registros da ultima ocorrência de um período específico"
    end_date: Data no formato YYYYMMDD
        start_date: Data no formato YYYYMMDD
        
    """
    url = f"https://awesomeapi-exchange.p.rapidapi.com/{format}/list/{moeda}"
    querystring = {}
    if end_date:
        querystring['end_date'] = end_date
    if start_date:
        querystring['start_date'] = start_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "awesomeapi-exchange.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

