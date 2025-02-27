import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def verificar_departamento(department: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Permite obtener los departamentos de colombia"
    
    """
    url = f"https://politicaldivitioncolombia.p.rapidapi.com/AKfycbzSTiwLZNeD_XdQuY51NLr1hrxtLESIESYcFYdkMz2UBZ6r7gkG--ENC5OXNXf9rmGm/exec"
    querystring = {'department': department, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "politicaldivitioncolombia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def verificar_departamento(department: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Permite obtener los departamentos de colombia"
    
    """
    url = f"https://politicaldivitioncolombia.p.rapidapi.com/AKfycbzSTiwLZNeD_XdQuY51NLr1hrxtLESIESYcFYdkMz2UBZ6r7gkG--ENC5OXNXf9rmGm/exec"
    querystring = {'department': department, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "politicaldivitioncolombia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

