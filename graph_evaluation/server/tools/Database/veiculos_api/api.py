import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def veiculo_tipo(veiculo_tipo: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retorna as marcas do tipo selecionado, os tipos disponíveis são "carros", "motos" e "caminhoes"."
    
    """
    url = f"https://veiculos-api.p.rapidapi.com/{veiculo_tipo}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "veiculos-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def veiculo_tipo_id_marca_id_modelo_id_modelo_ano(id_modelo_ano: str, veiculo_tipo: str, id_marca: str, id_modelo: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "/"
    
    """
    url = f"https://veiculos-api.p.rapidapi.com/{veiculo_tipo}/{id_marca}/{id_modelo}/{id_modelo_ano}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "veiculos-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def veiculo_tipo_id_marca(veiculo_tipo: str, id_marca: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retorna listagem dos veículos de uma determinada marca."
    
    """
    url = f"https://veiculos-api.p.rapidapi.com/{veiculo_tipo}/{id_marca}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "veiculos-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def veiculo_tipo_id_marca_id_modelo(id_marca: str, id_modelo: str, veiculo_tipo: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "/"
    
    """
    url = f"https://veiculos-api.p.rapidapi.com/{veiculo_tipo}/{id_marca}/{id_modelo}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "veiculos-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

