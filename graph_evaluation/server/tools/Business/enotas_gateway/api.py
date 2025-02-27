import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def buscar_empresas(pagenumber: int, pagesize: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Buscar empresas emissoras"
    pagenumber: Paginação, a partir de 0
        pagesize: Tamanho da página
        
    """
    url = f"https://enotas-enotas-gateway-v1.p.rapidapi.com/empresas"
    querystring = {'pageNumber': pagenumber, 'pageSize': pagesize, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "enotas-enotas-gateway-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

