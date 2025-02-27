import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_data_id_pelanggan_tagihan_pln_pasca_bayar(is_id: str='532411286263', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GET Data Pelanggan Tagihan PLN / PLN PASCA BAYAR /Postpaid
		- masukan id PLN di form id"
    
    """
    url = f"https://cek-id-pln-pasca-dan-pra-bayar.p.rapidapi.com/plnpostpaid/{is_id}"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cek-id-pln-pasca-dan-pra-bayar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_data_id_pelanggan_token_pln(is_id: str='56213840202', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GET Data Pelanggan Token PLN / PLN PRA BAYAR
		- masukan id PLN di form id"
    
    """
    url = f"https://cek-id-pln-pasca-dan-pra-bayar.p.rapidapi.com/pln/{is_id}/token_pln"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cek-id-pln-pasca-dan-pra-bayar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

