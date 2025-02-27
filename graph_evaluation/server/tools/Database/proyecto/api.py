import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def tienda(espada2: int=300, vida3: int=300, espada1: int=250, escudo2: int=150, espada3: int=350, vida2: int=250, escudo3: int=200, escudo1: int=100, vida1: int=200, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "La tienda del video juego"
    
    """
    url = f"https://proyecto.p.rapidapi.com/Tienda"
    querystring = {}
    if espada2:
        querystring['Espada2'] = espada2
    if vida3:
        querystring['Vida3'] = vida3
    if espada1:
        querystring['Espada1'] = espada1
    if escudo2:
        querystring['Escudo2'] = escudo2
    if espada3:
        querystring['Espada3'] = espada3
    if vida2:
        querystring['Vida2'] = vida2
    if escudo3:
        querystring['Escudo3'] = escudo3
    if escudo1:
        querystring['Escudo1'] = escudo1
    if vida1:
        querystring['Vida1'] = vida1
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "proyecto.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

