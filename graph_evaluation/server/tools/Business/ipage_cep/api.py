import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def address(cidade: str, uf: str, key: str, logradouro: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retuns the address from the state, city and street"
    cidade: City's name
        uf: Acronym of federative unit.Ex .: sp
        key: access key
        logradouro: Address you want to find
        
    """
    url = f"https://ipage_cep.p.rapidapi.com/ws/cep/v1/application/views/endereco/"
    querystring = {'cidade': cidade, 'uf': uf, 'key': key, 'logradouro': logradouro, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ipage_cep.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def zip_code(cep: str, key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the address from the entered zip code"
    cep: Brazilian zip code
        key: access key
        
    """
    url = f"https://ipage_cep.p.rapidapi.com/ws/cep/v1/application/views/cep/"
    querystring = {'cep': cep, 'key': key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ipage_cep.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cnpj(key: str, cnpj: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns address from company CNPJ"
    key: access key
        cnpj: Brazilian Company Registration Number
        
    """
    url = f"https://ipage_cep.p.rapidapi.com/ws/cep/v1/application/views/cnpj/"
    querystring = {'key': key, 'cnpj': cnpj, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ipage_cep.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def routes(key: str, cep: str, valor_gas: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find the time, distance and value in kilometers of the route between 3 postal codes (feature available on request)"
    key: Access key
        cep: Zip code origin, zip code destination 1, zip code destination2
        valor_gas: Value per KM
        
    """
    url = f"https://ipage_cep.p.rapidapi.com/ws/cep/v1/application/views/rota/"
    querystring = {'key': key, 'cep': cep, 'valor_gas': valor_gas, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ipage_cep.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def produto(key: str, code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns product data from its barcode"
    key: Enter your api key
        code: Enter the barcode
        
    """
    url = f"https://ipage_cep.p.rapidapi.com/ws/produto/v1/application/views/codebar/"
    querystring = {'key': key, 'code': code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ipage_cep.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

