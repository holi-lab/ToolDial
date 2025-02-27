import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def consulta_cpf_por_telefone(cpf: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Encontre o cpf pelo numero do telefone"
    
    """
    url = f"https://consulta-cpf-e-nome.p.rapidapi.com/BuscaCPFTelefone/{cpf}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "consulta-cpf-e-nome.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def consulta_por_cpf(cpf: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Informe o CPF para obter o resultado dos dados da pessoa fisica"
    
    """
    url = f"https://consulta-cpf-e-nome.p.rapidapi.com/BuscaCPF2/{cpf}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "consulta-cpf-e-nome.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def consulta_telefone(cpf: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Encontre o numero de telefone buscando pelo CPF da pessoa"
    
    """
    url = f"https://consulta-cpf-e-nome.p.rapidapi.com/BuscaTelefones/{cpf}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "consulta-cpf-e-nome.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def consulta_por_nome(nome: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Encontre um CPF apenas pelo nome completo da pessoa"
    
    """
    url = f"https://consulta-cpf-e-nome.p.rapidapi.com/BuscaNome/{nome}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "consulta-cpf-e-nome.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

