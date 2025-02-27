import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def cpf(cpf: str, nome: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Buscar pelo CPF do sócio ou administrador"
    nome: O nome é opcional, porém ao informa-lo melhoram as chances de encontrar os dados.
        
    """
    url = f"https://quadro-de-socios-cpf-cnpj.p.rapidapi.com/buscar-base.php"
    querystring = {'cpf': cpf, }
    if nome:
        querystring['nome'] = nome
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quadro-de-socios-cpf-cnpj.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cnpj(cnpj: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Buscar dados empresa pelo CNPJ"
    
    """
    url = f"https://quadro-de-socios-cpf-cnpj.p.rapidapi.com/buscar-base.php"
    querystring = {'cnpj': cnpj, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quadro-de-socios-cpf-cnpj.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cpfobliterado(nome: str, cpf: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Buscar quando o CPF está obliterado. Ex: ***053599**"
    nome: Nome do sócio ou administrador
        cpf: CPF obliterado
        
    """
    url = f"https://quadro-de-socios-cpf-cnpj.p.rapidapi.com/buscar-base.php"
    querystring = {'nome': nome, 'cpf': cpf, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quadro-de-socios-cpf-cnpj.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

