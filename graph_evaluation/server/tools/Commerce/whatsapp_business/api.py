import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def settings_business_about(content_type: str, authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Service that get Whatsapp Bussiness About."
    authorization: whatsapp_admin_api_token
        
    """
    url = f"https://whatsapp-business.p.rapidapi.com/settings/business/about"
    querystring = {'content-type': content_type, 'authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whatsapp-business.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def settings_business_photo(authorization: str, content_type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Service that get Whatsapp Bussiness profile photo."
    authorization: whatsapp_admin_api_token
        
    """
    url = f"https://whatsapp-business.p.rapidapi.com/settings/business/photo"
    querystring = {'authorization': authorization, 'content-type': content_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whatsapp-business.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def settings_business_profile(authorization: str, content_type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Service that get Whatsapp Bussiness settings."
    authorization: whatsapp_admin_api_token
        
    """
    url = f"https://whatsapp-business.p.rapidapi.com/settings/business/profile"
    querystring = {'authorization': authorization, 'content-type': content_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whatsapp-business.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

