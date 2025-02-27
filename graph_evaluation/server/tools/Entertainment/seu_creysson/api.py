import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def frase(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retorna uma citação famosa traduzida para o *Creyssonês*."
    
    """
    url = f"https://seu-creysson.p.rapidapi.com/frase"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seu-creysson.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def manxetes(limite: int=2, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retorna títulos e links para as últimas notícias da política, traduzidas para o idioma do *"Seu Creysson"*.
		
		**Fonte das notícias**
		**Portal G1: [https://g1.globo.com](url)**
		*Todos os direitos reservados*"
    limite: Limita o número de notícias que irá trazer. 

É restringido ao máximo de **10 notícias** por vez.
        
    """
    url = f"https://seu-creysson.p.rapidapi.com/manxetes"
    querystring = {}
    if limite:
        querystring['limite'] = limite
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seu-creysson.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

