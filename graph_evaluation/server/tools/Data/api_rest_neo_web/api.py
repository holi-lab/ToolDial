import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def clienteporid(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Mediante la solicitud "GET" se solicita la información del cliente requerido mediante la búsqueda de su ID."
    
    """
    url = f"https://api-rest-neo-web.p.rapidapi.com/cliente/7"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-rest-neo-web.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historialdeclientes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "## _Lista_
		
		Mediante la solicitud "GET" se solicita el listado de todos los clientes actuales en la Base de Datos."
    
    """
    url = f"https://api-rest-neo-web.p.rapidapi.com/cliente"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-rest-neo-web.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listadeproductos(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "## _Lista_
		
		Utilizando la solicitud "GET" se obtiene la lista de todos los productos y su información."
    
    """
    url = f"https://api-rest-neo-web.p.rapidapi.com/producto"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-rest-neo-web.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def productosvendidosaclientes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "## _Lista_
		
		Utilizando la solicitud "GET" obtenemos todos los productos que han sido vendidos y a que cliente se lo vendió.
		
		"
    
    """
    url = f"https://api-rest-neo-web.p.rapidapi.com/clienteproducto"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-rest-neo-web.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def productoporid(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "## _ID_
		
		Utilizando la solicitud "GET" y como parámetro un número que representa el ID del producto, se obtiene la información de dicho producto solicitado."
    
    """
    url = f"https://api-rest-neo-web.p.rapidapi.com/producto/3"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-rest-neo-web.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def productovendidoauncliente(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "## _Lista_
		
		Utilizando la solicitud "GET" obtenemos todos los productos que se vendieron a un solo cliente, utilizando el ID de ese mismo cliente."
    
    """
    url = f"https://api-rest-neo-web.p.rapidapi.com/clienteproducto/3"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-rest-neo-web.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

