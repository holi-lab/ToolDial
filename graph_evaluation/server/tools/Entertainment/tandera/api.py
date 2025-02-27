import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def filmes(titulo: str, tipo: str, pais: str='br', ano: int=2012, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Pesquisa de filmes nos provedores de streaming."
    titulo: O nome completo do filme.
        tipo: Informa se a pesquisa é de um filme ou série: mv ou tv.
        pais: País de consulta. Brasil é o país default. Países: Brasil (br) e Estados Unidos (us).
        ano: Ano de lançamento do filme.
        
    """
    url = f"https://tandera.p.rapidapi.com/"
    querystring = {'titulo': titulo, 'tipo': tipo, }
    if pais:
        querystring['pais'] = pais
    if ano:
        querystring['ano'] = ano
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tandera.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def s_ries(tipo: str, titulo: str, ano: int=None, pais: str='br', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Pesquisa de séries nos provedores de streaming."
    tipo: Informa se a pesquisa é de um filme ou série: mv ou tv.
        titulo: O nome completo da série.
        ano: Ano de lançamento da série.
        pais: País de consulta. Brasil é o país default. Países: Brasil (br) e Estados Unidos (us).
        
    """
    url = f"https://tandera.p.rapidapi.com/"
    querystring = {'tipo': tipo, 'titulo': titulo, }
    if ano:
        querystring['ano'] = ano
    if pais:
        querystring['pais'] = pais
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tandera.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

