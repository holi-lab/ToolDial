import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def compra(vida: str='["200","250","300"]', armas: str='["250","300","350"]', escudos: str='["100","150","200"]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Todos los artículos son utilizados en Origin"
    
    """
    url = f"https://origin1.p.rapidapi.com/Tienda/Compra"
    querystring = {}
    if vida:
        querystring['Vida'] = vida
    if armas:
        querystring['Armas'] = armas
    if escudos:
        querystring['Escudos'] = escudos
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "origin1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def venta(armas: str='["0","0","0"]', escudos: str='["0","0","0"]', vida: str='["0","0","0"]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Precios de venta en la tienda"
    
    """
    url = f"https://origin1.p.rapidapi.com/Tienda/Venta"
    querystring = {}
    if armas:
        querystring['Armas'] = armas
    if escudos:
        querystring['Escudos'] = escudos
    if vida:
        querystring['Vida'] = vida
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "origin1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

