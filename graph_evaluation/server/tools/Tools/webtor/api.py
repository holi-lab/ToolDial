import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def export_resource_content(resource_id: str, content_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Exports resource content.
		Provide urls to consume torrent content for download and streaming."
    
    """
    url = f"https://webtor.p.rapidapi.com/resource/{resource_id}/export/{content_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webtor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_resource_contents(resource_id: str, output: str=None, offset: int=0, path: str='/', limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists files and directories of specific resource.
		All ids in response can be used for export."
    
    """
    url = f"https://webtor.p.rapidapi.com/resource/{resource_id}/list"
    querystring = {}
    if output:
        querystring['output'] = output
    if offset:
        querystring['offset'] = offset
    if path:
        querystring['path'] = path
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webtor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_resource(resource_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Receives resource id and returns resource."
    
    """
    url = f"https://webtor.p.rapidapi.com/resource/{resource_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webtor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

