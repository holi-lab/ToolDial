import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def polygon_metadata_from_nft_smart_contract(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Greb metadata from given smart contract address"
    
    """
    url = f"https://nft12.p.rapidapi.com/bsc?address=0x3aEF41668fd611251Cb81bb9EE743A3C0af11aC1"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nft12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bsc_metadata_from_nft_smart_contract(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Greb metadata from given smart contract address"
    
    """
    url = f"https://nft12.p.rapidapi.com/bsc?address=0x3aEF41668fd611251Cb81bb9EE743A3C0af11aC1"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nft12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def eth_metadata_from_nft_smart_contract(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Greb metadata from given smart contract address"
    
    """
    url = f"https://nft12.p.rapidapi.com/eth?address=0x341a1c534248966c4b6afad165b98daed4b964ef"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nft12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

