import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def read_estado_list_estados_get(order_by: str='-id', objects_per_page: int=10, page: int=0, pagination: bool=None, limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get estado list.
		
		This will return a estados list to the user from the API."
    
    """
    url = f"https://mx-codigos-postales.p.rapidapi.com/estados/"
    querystring = {}
    if order_by:
        querystring['order_by'] = order_by
    if objects_per_page:
        querystring['objects_per_page'] = objects_per_page
    if page:
        querystring['page'] = page
    if pagination:
        querystring['pagination'] = pagination
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mx-codigos-postales.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def read_asentamiento_list_asentamientos_get(pagination: bool=None, order_by: str='-id', limit: int=10, objects_per_page: int=10, search: str=None, page: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get asentamiento list.
		
		This will return a asentamientos list to the user from the API."
    
    """
    url = f"https://mx-codigos-postales.p.rapidapi.com/asentamientos/"
    querystring = {}
    if pagination:
        querystring['pagination'] = pagination
    if order_by:
        querystring['order_by'] = order_by
    if limit:
        querystring['limit'] = limit
    if objects_per_page:
        querystring['objects_per_page'] = objects_per_page
    if search:
        querystring['search'] = search
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mx-codigos-postales.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def read_asentamiento_asentamientos_id_get(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get asentamiento by ID.
		
		This will return a asentamiento model by the id to the user from the API (only if it exists)."
    
    """
    url = f"https://mx-codigos-postales.p.rapidapi.com/asentamientos/{is_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mx-codigos-postales.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def read_estado_estados_id_get(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get estado by ID.
		
		This will return a estado model by the id to the user from the API (only if it exists)."
    
    """
    url = f"https://mx-codigos-postales.p.rapidapi.com/estados/{is_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mx-codigos-postales.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def read_municipio_municipios_id_get(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get municipio by ID.
		
		This will return a municipio model by the id to the user from the API (only if it exists)."
    
    """
    url = f"https://mx-codigos-postales.p.rapidapi.com/municipios/{is_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mx-codigos-postales.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def root_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://mx-codigos-postales.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mx-codigos-postales.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def read_ciudad_ciudades_id_get(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get ciudad by ID.
		
		This will return a ciudad model by the id to the user from the API (only if it exists)."
    
    """
    url = f"https://mx-codigos-postales.p.rapidapi.com/ciudades/{is_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mx-codigos-postales.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def read_municipio_list_municipios_get(limit: int=10, objects_per_page: int=10, pagination: bool=None, order_by: str='-id', page: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get municipio list.
		
		This will return a municipios list to the user from the API."
    
    """
    url = f"https://mx-codigos-postales.p.rapidapi.com/municipios/"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if objects_per_page:
        querystring['objects_per_page'] = objects_per_page
    if pagination:
        querystring['pagination'] = pagination
    if order_by:
        querystring['order_by'] = order_by
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mx-codigos-postales.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def read_ciudad_list_ciudades_get(order_by: str='-id', search: str=None, pagination: bool=None, limit: int=10, page: int=0, objects_per_page: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get ciudad list.
		
		This will return a ciudades list to the user from the API."
    
    """
    url = f"https://mx-codigos-postales.p.rapidapi.com/ciudades/"
    querystring = {}
    if order_by:
        querystring['order_by'] = order_by
    if search:
        querystring['search'] = search
    if pagination:
        querystring['pagination'] = pagination
    if limit:
        querystring['limit'] = limit
    if page:
        querystring['page'] = page
    if objects_per_page:
        querystring['objects_per_page'] = objects_per_page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mx-codigos-postales.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

