import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def resultados_euromillones(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Devuelve los resultados del sorteo de Euromillones para un día en concreto."
    
    """
    url = f"https://loterias-y-apuestas-del-estado.p.rapidapi.com/v1/loteria/resultados/euromillones"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "loterias-y-apuestas-del-estado.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def resultados_la_primitiva(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Devuelve los resultados del sorteo de la primitiva para un día en concreto."
    
    """
    url = f"https://loterias-y-apuestas-del-estado.p.rapidapi.com/v1/loteria/resultados/primitiva"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "loterias-y-apuestas-del-estado.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def resultados_quinigol(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Devuelve los resultados del sorteo de Bonoloto para un día en concreto."
    
    """
    url = f"https://loterias-y-apuestas-del-estado.p.rapidapi.com/v1/loteria/resultados/quinigol"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "loterias-y-apuestas-del-estado.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def resultados_bonoloto(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Devuelve los resultados del sorteo de Bonoloto para un día en concreto."
    
    """
    url = f"https://loterias-y-apuestas-del-estado.p.rapidapi.com/v1/loteria/resultados/bonoloto"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "loterias-y-apuestas-del-estado.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def resultados_lototurf(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Devuelve los resultados del sorteo de Bonoloto para un día en concreto."
    
    """
    url = f"https://loterias-y-apuestas-del-estado.p.rapidapi.com/v1/loteria/resultados/lototurf"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "loterias-y-apuestas-del-estado.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def resultados_loter_a_nacional(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Devuelve los resultados del sorteo de Bonoloto para un día en concreto."
    
    """
    url = f"https://loterias-y-apuestas-del-estado.p.rapidapi.com/v1/loteria/resultados/loteria-nacional"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "loterias-y-apuestas-del-estado.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def resultados_el_gordo(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Devuelve los resultados del sorteo de Bonoloto para un día en concreto."
    
    """
    url = f"https://loterias-y-apuestas-del-estado.p.rapidapi.com/v1/loteria/resultados/el-gordo"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "loterias-y-apuestas-del-estado.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def resultados_quintuple_plus(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Devuelve los resultados del sorteo de Bonoloto para un día en concreto."
    
    """
    url = f"https://loterias-y-apuestas-del-estado.p.rapidapi.com/v1/loteria/resultados/quintuple-plus"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "loterias-y-apuestas-del-estado.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def resultados_la_quiniela(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Devuelve los resultados del sorteo de Bonoloto para un día en concreto."
    
    """
    url = f"https://loterias-y-apuestas-del-estado.p.rapidapi.com/v1/loteria/resultados/la-quiniela"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "loterias-y-apuestas-del-estado.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

