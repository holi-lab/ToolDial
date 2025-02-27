import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def feriados_dos_estados(estado: str, ano: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retorna os feriados de certo estado. Os parâmetros necessários são: 1. Estado - Sigla da Unidade Federativa em letra maíuscula. 2. Ano"
    
    """
    url = f"https://feriados-brasileiros1.p.rapidapi.com/read_uf"
    querystring = {'estado': estado, 'ano': ano, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "feriados-brasileiros1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def feriados_das_cidades(cidade: str, ano: str, estado: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retorna os feriados de certa cidade brasileira. Os parâmetros necessários são: 1. Cidade - Nome da cidade sem acentos ou símbolos especiais, com exceção do hífen ("-") que deve ser mantido. 2. Estado - Unidade Federativa correspondente à cidade. Usar a sigla e em letra maíuscula. 3. Ano"
    
    """
    url = f"https://feriados-brasileiros1.p.rapidapi.com/read"
    querystring = {'cidade': cidade, 'ano': ano, 'estado': estado, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "feriados-brasileiros1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

