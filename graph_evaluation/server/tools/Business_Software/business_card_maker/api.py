import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def api_business_card_fromat(website: str, lastname: str, firstname: str, company: str, template: str, phone: str, email: str, fromat: str, position: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Business Card."
    template: /api/templates for getting template list
        fromat: Response data format.
**formats:**

- json
- html
- jpg
- png
        
    """
    url = f"https://business-card-maker.p.rapidapi.com/api/business-card/{fromat}"
    querystring = {'webSite': website, 'lastName': lastname, 'firstName': firstname, 'company': company, 'template': template, 'phone': phone, 'email': email, 'position': position, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "business-card-maker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_templates(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get business cards templates list."
    
    """
    url = f"https://business-card-maker.p.rapidapi.com/api/templates"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "business-card-maker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get version number."
    
    """
    url = f"https://business-card-maker.p.rapidapi.com/api"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "business-card-maker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

