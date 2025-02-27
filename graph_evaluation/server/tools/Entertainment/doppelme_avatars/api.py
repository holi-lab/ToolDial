import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def list_assets(bodytype_id: int, assettype: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of assets available given a bodytype and assettype"
    bodytype_id: Body type identification number (e.g. 1101)
        assettype: Asset type (e.g. top)
        
    """
    url = f"https://doppelme-avatars.p.rapidapi.com/assets/{bodytype_id}/{assettype}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "doppelme-avatars.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_asset_types(bodytype_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return list of asset types that are available for the given body type identifier."
    bodytype_id: Body type identification number (e.g. 1101)
        
    """
    url = f"https://doppelme-avatars.p.rapidapi.com/assets/{bodytype_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "doppelme-avatars.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_body_types(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of avatar body types that are available to choose when creating your avatar"
    
    """
    url = f"https://doppelme-avatars.p.rapidapi.com/bodytypes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "doppelme-avatars.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

