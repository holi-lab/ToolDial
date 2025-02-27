import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def green_journal(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A Revista Verde (RV) é periódico científico internacional semestral do Programa Escola Verde e do Grupo de Pesquisa em Educação Ambiental Interdisciplinar (CNPq) sobre problemáticas socioambientais e sustentabilidade.
		
		Website: https://revistaverde.escolaverde.org"
    
    """
    url = f"https://revista-verde.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "revista-verde.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

