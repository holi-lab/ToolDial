import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def make_id(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Creates Unique address, private and public key pair. The backend process uses Bitcoin new wallet structure to create IDs. IDs have the following structure. Feel free to use your own IDs or have one created for you.
		
		{
		  "address":"1EqyJUSZyb4ygzs74PEawutSztbh421QB5"
		  "privateKey":"KyonVn97Gvddn1r3ztey2ad74MmKtuXaQRPpBfaruxBWQpdggved"
		  "publicKey":"03cebdded554cbca8173903173e15df1af5297256b58c0ebedd8850091774fc29b"
		}"
    
    """
    url = f"https://datapi1.p.rapidapi.com/id"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "datapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def make_uuid(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Creates unique UUID. The backend process uses Bitcoin new wallet to create IDs. Bitcoin public + private key combo are used to create unique UUID. 
		
		IDs have the following structure. Feel free to use your own IDs or have one created for you.
		
		{ "uuid":"526901e8-be33-5a43-84e8-1a97d9de87c7" }"
    
    """
    url = f"https://datapi1.p.rapidapi.com/uuid"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "datapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_value(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Value"
    
    """
    url = f"https://datapi1.p.rapidapi.com/get/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "datapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

