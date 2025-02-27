import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def resultado_da_lotomania(numeroconcurso: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retorna o resultado da Lotomania dado um número de concurso"
    
    """
    url = f"https://loterias-caixa.p.rapidapi.com/lotomania/{numeroconcurso}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "loterias-caixa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def resultado_da_quina(numeroconcurso: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retorna o resultado da Quina dado um número de concurso"
    
    """
    url = f"https://loterias-caixa.p.rapidapi.com/quina/{numeroconcurso}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "loterias-caixa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def resultado_da_loteca(numeroconcurso: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retorna o resultado da Loteca dado um número de concurso"
    
    """
    url = f"https://loterias-caixa.p.rapidapi.com/loteca/{numeroconcurso}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "loterias-caixa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def resultado_da_federal(numeroconcurso: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retorna o resultado da Federal dado um número de concurso"
    
    """
    url = f"https://loterias-caixa.p.rapidapi.com/federal/{numeroconcurso}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "loterias-caixa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def resultado_da_super_sete(numeroconcurso: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retorna o resultado da Super Sete dado um número de concurso"
    
    """
    url = f"https://loterias-caixa.p.rapidapi.com/supersete/{numeroconcurso}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "loterias-caixa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def resultado_da_dia_de_sorte(numeroconcurso: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retorna o resultado da Dia de Sorte dado um número de concurso"
    
    """
    url = f"https://loterias-caixa.p.rapidapi.com/diadesorte/{numeroconcurso}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "loterias-caixa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def resultado_da_timemania(numeroconcurso: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retorna o resultado da Timemania dado um número de concurso"
    
    """
    url = f"https://loterias-caixa.p.rapidapi.com/timemania/{numeroconcurso}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "loterias-caixa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def resultado_da_dupla_sena(numeroconcurso: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retorna o resultado da Dupla Sena dado um número de concurso"
    
    """
    url = f"https://loterias-caixa.p.rapidapi.com/duplasena/{numeroconcurso}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "loterias-caixa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def resultado_da_lotof_cil(numeroconcurso: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retorna o resultado da Lotofácil dado um número de concurso"
    
    """
    url = f"https://loterias-caixa.p.rapidapi.com/lotofacil/{numeroconcurso}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "loterias-caixa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def resultado_da_megasena(numeroconcurso: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retorna o resultado da Megasena dado um número de concurso"
    
    """
    url = f"https://loterias-caixa.p.rapidapi.com/megasena/{numeroconcurso}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "loterias-caixa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

