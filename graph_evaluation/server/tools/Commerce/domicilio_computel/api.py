import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def cp(cp: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "INGRESA EL CODIGO POSTAL"
    cp: Es el codigo postal mexico
        
    """
    url = f"https://domicilio-computel.p.rapidapi.com/code_postal/consulta/cp.php"
    querystring = {'cp': cp, }
    if cp:
        querystring['cp'] = cp
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "domicilio-computel.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

