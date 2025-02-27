import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def lista_de_municipios_por_provincia(id_provincia: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lista todas lps municipios  de una provincia"
    
    """
    url = f"https://organizacion-territorial-de-espana.p.rapidapi.com/provincia/{id_provincia}/municipio"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "organizacion-territorial-de-espana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lista_de_provincias_por_comunidad_aut_noma(id_comunidad: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lista todas las provincias de una Comunidad Autónoma"
    
    """
    url = f"https://organizacion-territorial-de-espana.p.rapidapi.com/comunidad/{id_comunidad}/provincia"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "organizacion-territorial-de-espana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lista_de_provincias(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lista el nombre y el id de todas las provincias de España"
    
    """
    url = f"https://organizacion-territorial-de-espana.p.rapidapi.com/provincia"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "organizacion-territorial-de-espana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lista_de_comunidades_aut_nomas(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lista el nombre y el id de todas las Comunidades Autónomas de España"
    
    """
    url = f"https://organizacion-territorial-de-espana.p.rapidapi.com/comunidad"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "organizacion-territorial-de-espana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

