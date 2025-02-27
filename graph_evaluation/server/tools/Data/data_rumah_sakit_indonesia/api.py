import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_rumah_sakit_by_kabkota(kabkota: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Menampilkan Data Rumah Sakit Indonesia Berdasarkan Kab/Kota"
    kabkota: Masukan nama kab/kota yang ingin ditampilkan
        
    """
    url = f"https://data-rumah-sakit-indonesia.p.rapidapi.com/"
    querystring = {'kabkota': kabkota, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "data-rumah-sakit-indonesia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_rumah_sakit(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Menampilkan Semua Data Rumah Sakit Indonesia Per Kab/Kota"
    
    """
    url = f"https://data-rumah-sakit-indonesia.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "data-rumah-sakit-indonesia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

