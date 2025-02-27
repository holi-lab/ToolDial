import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def winner_prestasi_les_privat_calistung_sd_smp_sma_sbmptn_simakui_akm_osn_di_jabodetabek_indonesia_copy(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Winner Prestasi Les Privat Calistung Sd Smp SMA SBMPTN SimakUI AKM OSN di Jabodetabek ~ Indonesia"
    
    """
    url = f"https://les-privat-calistung.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "les-privat-calistung.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def winner_prestasi_les_privat_calistung_sd_smp_sma_sbmptn_simakui_akm_osn_di_jabodetabek_indonesia(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Winner Prestasi Les Privat Calistung Sd Smp SMA SBMPTN SimakUI AKM OSN di Jabodetabek ~ Indonesia"
    
    """
    url = f"https://les-privat-calistung.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "les-privat-calistung.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

