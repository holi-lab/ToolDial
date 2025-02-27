import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def new_send_sms(sendmsg: str, email: str, sendto: str, password: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "New Send SMS"
    
    """
    url = f"https://bulk-sms-sending-sri-lanka.p.rapidapi.com/"
    querystring = {'sendmsg': sendmsg, 'email': email, 'sendto': sendto, 'password': password, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bulk-sms-sending-sri-lanka.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def account_details(email: str, password: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Account Details"
    email: Enter the email id provided to create an account with us
        password: Enter the password you provided when you create the account
        
    """
    url = f"https://bulk-sms-sending-sri-lanka.p.rapidapi.com/"
    querystring = {'email': email, 'password': password, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bulk-sms-sending-sri-lanka.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sms_credit_balance(email: str, password: str, bal: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter Your Account Email and Password To Know Your SMS Credit Balance."
    
    """
    url = f"https://bulk-sms-sending-sri-lanka.p.rapidapi.com/"
    querystring = {'email': email, 'password': password, }
    if bal:
        querystring['bal'] = bal
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bulk-sms-sending-sri-lanka.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

