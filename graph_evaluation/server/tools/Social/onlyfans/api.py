import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def list(sort: str, signend: str, signstart: str, apptoken: str, xbc: str, timezone: str, useragent: str, field: str, offset: str, sess: str, auth_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Used to get a full list of expired user details
		
		Must hit the auth endpoint first!
		Also best to hit the count endpoint beforehand."
    offset: Must be divisible by 200
        
    """
    url = f"https://onlyfans.p.rapidapi.com/expired/details/"
    querystring = {'sort': sort, 'signend': signend, 'signstart': signstart, 'apptoken': apptoken, 'xbc': xbc, 'timezone': timezone, 'useragent': useragent, 'field': field, 'offset': offset, 'sess': sess, 'auth_id': auth_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "onlyfans.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def performer_model_info(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detail info about a performer / model"
    username: Can be a userid or a username
        
    """
    url = f"https://onlyfans.p.rapidapi.com/performer/"
    querystring = {'username': username, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "onlyfans.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def payouts(apptoken: str, signend: str, signstart: str, xbc: str, timezone: str, useragent: str, sess: str, auth_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Used to get the last 100 payouts
		
		Must hit the auth endpoint first!"
    
    """
    url = f"https://onlyfans.p.rapidapi.com/statements/payouts/"
    querystring = {'apptoken': apptoken, 'signend': signend, 'signstart': signstart, 'xbc': xbc, 'timezone': timezone, 'useragent': useragent, 'sess': sess, 'auth_id': auth_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "onlyfans.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def earnings(apptoken: str, xbc: str, signstart: str, signend: str, auth_id: str, sess: str, useragent: str, timezone: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Used to get the last 100 earnings
		
		Must hit the auth endpoint first!"
    
    """
    url = f"https://onlyfans.p.rapidapi.com/statements/earnings/"
    querystring = {'apptoken': apptoken, 'xbc': xbc, 'signstart': signstart, 'signend': signend, 'auth_id': auth_id, 'sess': sess, 'useragent': useragent, 'timezone': timezone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "onlyfans.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def authentication(signstart: str, apptoken: str, timezone: str, signend: str, xbc: str, sess: str, useragent: str, auth_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Best to call the Sign Info first take those values and pass them on"
    signstart: Value from /signinfo/ signinfo.start
        apptoken: Value from /signinfo/
        signend: Value from /signinfo/ signinfo.start
        xbc: Value from localstorage.bcTokenSha
        sess: Value from cookie.sess
        auth_id: Value from cookie.auth_id
        
    """
    url = f"https://onlyfans.p.rapidapi.com/auth/"
    querystring = {'signstart': signstart, 'apptoken': apptoken, 'timezone': timezone, 'signend': signend, 'xbc': xbc, 'sess': sess, 'useragent': useragent, 'auth_id': auth_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "onlyfans.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sign_info(useragent: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get required up to date sign info"
    
    """
    url = f"https://onlyfans.p.rapidapi.com/signinfo/"
    querystring = {'useragent': useragent, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "onlyfans.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mass_messages(timezone: str, useragent: str, auth_id: str, signstart: str, signend: str, sess: str, xbc: str, apptoken: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Used to get the last 100 mass messages
		
		Must hit the auth endpoint first!"
    
    """
    url = f"https://onlyfans.p.rapidapi.com/mass/messages/"
    querystring = {'timezone': timezone, 'useragent': useragent, 'auth_id': auth_id, 'signstart': signstart, 'signend': signend, 'sess': sess, 'xbc': xbc, 'apptoken': apptoken, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "onlyfans.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def count(signstart: str, apptoken: str, signend: str, xbc: str, timezone: str, useragent: str, auth_id: str, sess: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Used to get the total expired followers
		
		Must hit the auth endpoint first!"
    
    """
    url = f"https://onlyfans.p.rapidapi.com/expired/count/"
    querystring = {'signstart': signstart, 'apptoken': apptoken, 'signend': signend, 'xbc': xbc, 'timezone': timezone, 'useragent': useragent, 'auth_id': auth_id, 'sess': sess, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "onlyfans.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

