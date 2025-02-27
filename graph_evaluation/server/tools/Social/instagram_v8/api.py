import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_account_info(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves json of instagram account info by username"
    
    """
    url = f"https://instagram74.p.rapidapi.com/account-info"
    querystring = {'username': username, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram74.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_account_posts(userid: int, after: str='QVFCYWI2WDlNS2luREg1d040d1pjYlJlRVRjWUpBSjljWDNpZGpodms1VmpBUUtfSGtPWFd6Vm5lV2k4bkFDMkh1bmpBa19IX3RaY1VLSjZ6NFo1R0kxVA==', first: int=12, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get instagram medias by username id."
    after: Cursor after which the medias should be loaded. Technically this serves as \"offset\". Get current cursor via response.pageinfo.endcursor (see example response)
        first: default value 12
        
    """
    url = f"https://instagram74.p.rapidapi.com/account-medias"
    querystring = {'userid': userid, }
    if after:
        querystring['after'] = after
    if first:
        querystring['first'] = first
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram74.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

