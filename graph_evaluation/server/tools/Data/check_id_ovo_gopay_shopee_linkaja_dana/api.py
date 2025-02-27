import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_shopeepay_data(is_id: str='081357645086', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "input phone number ShopeePay"
    
    """
    url = f"https://check-id-ovo-gopay-shopee-linkaja-dana.p.rapidapi.com/cek_ewallet/{is_id}/shopeepay"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "check-id-ovo-gopay-shopee-linkaja-dana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_linkaja_data(is_id: str='081223222224', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "input phone number Linkaja"
    
    """
    url = f"https://check-id-ovo-gopay-shopee-linkaja-dana.p.rapidapi.com/cek_ewallet/{is_id}/linkaja"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "check-id-ovo-gopay-shopee-linkaja-dana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_kaspro_data(is_id: str='081223222224', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "input phone number KasPro"
    
    """
    url = f"https://check-id-ovo-gopay-shopee-linkaja-dana.p.rapidapi.com/cekewallet/{is_id}/KASPRO"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "check-id-ovo-gopay-shopee-linkaja-dana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_i_saku_data(is_id: str='081223222224', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "input phone number i.Saku"
    
    """
    url = f"https://check-id-ovo-gopay-shopee-linkaja-dana.p.rapidapi.com/cekewallet/{is_id}/ISAKU"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "check-id-ovo-gopay-shopee-linkaja-dana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_doku_data(is_id: str='1098222224', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "input id number Doku Wallet"
    
    """
    url = f"https://check-id-ovo-gopay-shopee-linkaja-dana.p.rapidapi.com/cekewallet/{is_id}/DOKU"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "check-id-ovo-gopay-shopee-linkaja-dana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_gopay_data(is_id: str='081223222224', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "input phone number Gopay"
    
    """
    url = f"https://check-id-ovo-gopay-shopee-linkaja-dana.p.rapidapi.com/cek_ewallet/{is_id}/gopay"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "check-id-ovo-gopay-shopee-linkaja-dana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_ovo_data(is_id: str='081223222224', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "input phone number OVO"
    
    """
    url = f"https://check-id-ovo-gopay-shopee-linkaja-dana.p.rapidapi.com/cek_ewallet/{is_id}/ovo"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "check-id-ovo-gopay-shopee-linkaja-dana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_dana_data(is_id: str='081223222224', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "input phone number DANA"
    
    """
    url = f"https://check-id-ovo-gopay-shopee-linkaja-dana.p.rapidapi.com/cek_ewallet/{is_id}/dana"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "check-id-ovo-gopay-shopee-linkaja-dana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

