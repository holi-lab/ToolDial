import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def company_api_streaming(domain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "An alternative to using our async lookup or webhooks is our streaming API. This API is ideal if you don’t mind long lived requests (e.g. you’re making requests to BigPicture from a messaging queue)."
    
    """
    url = f"https://company5.p.rapidapi.com/companies/find/stream"
    querystring = {'domain': domain, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "company5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def company_api(domain: str, webhookid: str='12345', webhookurl: str='https://requestb.in/wpyz2mwp', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Company API enables you to lookup company data based on a domain name. This can be used to retrieve the company's information such as name, address, industry, headcount, and various other details from their domain name."
    
    """
    url = f"https://company5.p.rapidapi.com/companies/find"
    querystring = {'domain': domain, }
    if webhookid:
        querystring['webhookId'] = webhookid
    if webhookurl:
        querystring['webhookUrl'] = webhookurl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "company5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

