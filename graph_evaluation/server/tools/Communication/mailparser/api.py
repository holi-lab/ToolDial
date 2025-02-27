import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def messages_id_parts_partid(is_id: str, partid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve individual message part"
    
    """
    url = f"https://activecom-smtpio-v1.p.rapidapi.com/messages/{is_id}/parts/{partid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "activecom-smtpio-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def messages_id_structure(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Succesfully retrieved resource"
    
    """
    url = f"https://activecom-smtpio-v1.p.rapidapi.com/messages/{is_id}/structure"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "activecom-smtpio-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def messages_id_parts(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve message parts information"
    
    """
    url = f"https://activecom-smtpio-v1.p.rapidapi.com/messages/{is_id}/parts"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "activecom-smtpio-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def messages_id_subject(is_id: str, raw: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve message subject"
    raw: Return without formatting. Default: false
        
    """
    url = f"https://activecom-smtpio-v1.p.rapidapi.com/messages/{is_id}/subject"
    querystring = {}
    if raw:
        querystring['raw'] = raw
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "activecom-smtpio-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def messages_id_headers(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve message headers"
    
    """
    url = f"https://activecom-smtpio-v1.p.rapidapi.com/messages/{is_id}/headers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "activecom-smtpio-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def messages_id(is_id: str, replace_inline_url_path: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve message body. If it is a multipart message and if html type is available, html version will be returned."
    replace_inline_url_path: Replace cid:content_id of inline attachments with the correct url to get them through this api. Default: true
        
    """
    url = f"https://activecom-smtpio-v1.p.rapidapi.com/messages/{is_id}"
    querystring = {}
    if replace_inline_url_path:
        querystring['replace_inline_url_path'] = replace_inline_url_path
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "activecom-smtpio-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

