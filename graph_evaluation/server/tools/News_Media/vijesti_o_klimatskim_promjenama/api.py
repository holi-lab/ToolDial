import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def preuzmi_vijesti_o_klimatskim_promijenama_s_odre_enog_izvora(newspaperid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Ovaj endpint vraća vijesti o klimatskim promijenama s točno određenog unesenog izvora"
    
    """
    url = f"https://vijesti-o-klimatskim-promjenama.p.rapidapi.com/news/{newspaperid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vijesti-o-klimatskim-promjenama.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def preuzmi_sve_vijesti_o_klimatskim_promjenama(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Ovaj endpoint će vratiti vijesti o klimatskim promjenama sa svih dohvaćenih izvora"
    
    """
    url = f"https://vijesti-o-klimatskim-promjenama.p.rapidapi.com/news"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vijesti-o-klimatskim-promjenama.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

