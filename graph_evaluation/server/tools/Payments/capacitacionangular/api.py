import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def productos(nombre: str=None, precio: str=None, cantidadstock: str=None, is_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Productos prueba"
    
    """
    url = f"https://capacitacionangular.p.rapidapi.com/productos"
    querystring = {}
    if nombre:
        querystring['Nombre'] = nombre
    if precio:
        querystring['Precio'] = precio
    if cantidadstock:
        querystring['CantidadStock'] = cantidadstock
    if is_id:
        querystring['Id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "capacitacionangular.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categorias(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Categorias"
    
    """
    url = f"https://capacitacionangular.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "capacitacionangular.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cliente(is_id: int=None, edad: str=None, celular: str=None, cedula: str=None, nombre: str=None, direccion: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Clientes"
    
    """
    url = f"https://capacitacionangular.p.rapidapi.com/Cliente"
    querystring = {}
    if is_id:
        querystring['Id'] = is_id
    if edad:
        querystring['Edad'] = edad
    if celular:
        querystring['Celular'] = celular
    if cedula:
        querystring['Cedula'] = cedula
    if nombre:
        querystring['Nombre'] = nombre
    if direccion:
        querystring['Direccion'] = direccion
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "capacitacionangular.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

