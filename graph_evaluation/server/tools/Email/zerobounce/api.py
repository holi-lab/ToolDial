import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def activity_data(api_key: str, email: str, ip_address: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Our Activity Data feature allows you to gather insights into your subscribers’ overall email engagement. The tool returns data regarding opens, clicks, forwards and unsubscribes that have taken place in the past 30, 90, 180 or 365 days. Thus, you can improve your targeting and personalization, and run more successful email campaigns."
    api_key: Your API Key, found in your account.
        email: The email address you want to check activity for
        ip_address: The IP Address of where the Email signed up from - You can pass in blank
        
    """
    url = f"https://zerobounce1.p.rapidapi.com/v1/activity"
    querystring = {'api_key': api_key, 'email': email, }
    if ip_address:
        querystring['ip_address'] = ip_address
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zerobounce1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def validate_email_with_ip_v1(ipaddress: str, email: str, apikey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Email Validation Endpoint with IP for Geolocation"
    ipaddress: The IP Address the email signed up from (Can be blank, but parameter required)
        email: The email address you want to validate

        apikey: Your API Key, found in your account

        
    """
    url = f"https://zerobounce1.p.rapidapi.com/v1/validatewithip"
    querystring = {'ipaddress': ipaddress, 'email': email, 'apikey': apikey, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zerobounce1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def validate_v1(email: str, apikey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Email Validation Endpoint"
    email: The email address you want to validate
        apikey: Your API Key, found in your account.
        
    """
    url = f"https://zerobounce1.p.rapidapi.com/v1/validate"
    querystring = {'email': email, 'apikey': apikey, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zerobounce1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def validate(api_key: str, email: str, ip_address: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Validates Email"
    api_key: This is the API KEY located in the ZeroBounce Members Section of the website.
        email: The email address you want to validate
        ip_address: The IP Address the email signed up from - You can pass in blank
        
    """
    url = f"https://zerobounce1.p.rapidapi.com/v2/validate"
    querystring = {'api_key': api_key, 'email': email, }
    if ip_address:
        querystring['ip_address'] = ip_address
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zerobounce1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

