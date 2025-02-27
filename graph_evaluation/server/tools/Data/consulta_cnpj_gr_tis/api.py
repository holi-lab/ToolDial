import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def consulta_unificada(cnpj: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retorna todos os dados presentes no Comprovante de Inscrição do Cadastro Nacional da Pessoa Jurídica junto a Receita Federal"
    cnpj: CNPJ da Empresa
        
    """
    url = f"https://consulta-cnpj-gratis.p.rapidapi.com/companies/{cnpj}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "consulta-cnpj-gratis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def consulta_estabelecimento(cnpj: str, simples: bool=None, registrations: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Consulta Receita Federal, Simples Nacional e Cadastro de Contribuintes através do CNPJ"
    simples: Adiciona informações do Simples Nacional
        registrations: UFs separadas por vírgula para adicionar informações do Cadastro de Contribuintes, utilize 'BR' para considerar todas.
        
    """
    url = f"https://consulta-cnpj-gratis.p.rapidapi.com/office/{cnpj}"
    querystring = {}
    if simples:
        querystring['simples'] = simples
    if registrations:
        querystring['registrations'] = registrations
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "consulta-cnpj-gratis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

