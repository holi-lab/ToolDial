import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_account_info(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get general account information like the email of the account owner and available credits."
    
    """
    url = f"https://blaze-verify.p.rapidapi.com/v1/account"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blaze-verify.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_status_of_a_batch(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GET requests to the batch endpoint will get the current status of the batch verification job specified in the "id" parameter.<br><br>When a credit card transaction is necessary to obtain enough credits to verify a batch, billing related messages will be returned if there is an error. These will be sent with a 402 response code.<br><br>"
    id: The id of the batch.
        
    """
    url = f"https://blaze-verify.p.rapidapi.com/v1/batch"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blaze-verify.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def verify_an_email(email: str, accept_all: bool=None, smtp: bool=None, timeout: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Verify a single email. If a verification request takes longer than the timeout, you may retry this request for up to 5 minutes. After 5 minutes, further requests will count against your usage. The verification result will be returned when it is available.<br><br>"
    email: The email you want verified.
        accept_all: Does an accept-all check. Heavily impacts API's response time. Default: false
        smtp: The SMTP step takes up a majority of the API's response time. If you would like to speed up your response times, you can disable this step. Default: true
        timeout: Optional timeout to wait for response (in seconds). Min: 2, Max: 30. Default: 5
        
    """
    url = f"https://blaze-verify.p.rapidapi.com/v1/verify"
    querystring = {'email': email, }
    if accept_all:
        querystring['accept_all'] = accept_all
    if smtp:
        querystring['smtp'] = smtp
    if timeout:
        querystring['timeout'] = timeout
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blaze-verify.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def heartbeat(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns API Status"
    
    """
    url = f"https://blaze-verify.p.rapidapi.com/v1/heartbeat"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blaze-verify.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

