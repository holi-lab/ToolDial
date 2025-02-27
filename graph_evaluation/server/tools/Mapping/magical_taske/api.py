import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def fetch_details_of_multiple_records(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint fetches details of more than 1 record. Make sure you pass a valid record ids. If you pass in an invalid id, you will get an `Internal Sever Error 500` for the whole request.
		
		You can request details of a maximum of 10 records at a go in this endpoint. The `id`s should be separated using a comma with no spaces."
    id: Record id
        
    """
    url = f"https://magical-taske.p.rapidapi.com/details"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "magical-taske.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_by_type_and_region(type: str, limit: int, region: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can make a request finer by combining region and type. This endpoint will return the records requested. You have to provide a limit of the number of records you want."
    type: Type of record you want. Refer to README above to see accepted types.
        
    """
    url = f"https://magical-taske.p.rapidapi.com/"
    querystring = {'type': type, 'limit': limit, 'region': region, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "magical-taske.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_by_region(region: str, limit: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint fetches records by region as defined in the `ABOUT` section above. You have to provide region and a limit of the number of records returned."
    region: Type of record you want. Refer to README above to see accepted types.
        
    """
    url = f"https://magical-taske.p.rapidapi.com/"
    querystring = {'region': region, 'limit': limit, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "magical-taske.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_by_type(type: str, limit: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint fetches records by type as defined in `ABOUT` section above. You have to provide type and a limit of the number of records returned."
    type: Type of record you want. Refer to README above to see accepted types.
        
    """
    url = f"https://magical-taske.p.rapidapi.com/"
    querystring = {'type': type, 'limit': limit, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "magical-taske.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

