import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def download_merged_pdf(uuid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Download pdf after merging"
    uuid: The uuid of the virtual pdf
        
    """
    url = f"https://pdf-fusion-plus.p.rapidapi.com/api/multi_merge_pdf/{uuid}/get_pdf"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pdf-fusion-plus.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def virtual_pdf_details(uuid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Show virtual PDF status. You can see attached pages and status"
    uuid: The uuid of the virtual pdf
        
    """
    url = f"https://pdf-fusion-plus.p.rapidapi.com/api/multi_merge_pdf/{uuid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pdf-fusion-plus.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

