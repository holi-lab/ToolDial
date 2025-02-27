import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def consulta_por_cnpj(cnpj: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Consulta dados e informações do CNPJ, a demanda de informações está sempre sendo atualizada para melhor atende-los **"
    
    """
    url = f"https://consulta-cpf-cnpj-brasil.p.rapidapi.com/cnpj/{cnpj}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "consulta-cpf-cnpj-brasil.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def consulta_dados_societ_rios(id_socio: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**## consulta dados dos sócios do cnpj consultado pelo id_socio**"
    
    """
    url = f"https://consulta-cpf-cnpj-brasil.p.rapidapi.com/socio/{id_socio}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "consulta-cpf-cnpj-brasil.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def consulta_por_cpf(cpf: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Consulta por CPF endpoint**"
    
    """
    url = f"https://consulta-cpf-cnpj-brasil.p.rapidapi.com/cpf/{cpf}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "consulta-cpf-cnpj-brasil.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

