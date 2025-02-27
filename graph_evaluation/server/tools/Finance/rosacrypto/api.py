import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_8_gettricker(clientsecert: str, content_type: str, clientid: str, netsessionid: str='c2fEErYaPUAKfOGE4zD15Mh2Nh0=', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://rosacrypto1.p.rapidapi.com/data/ticker/searchAll"
    querystring = {'clientSecert': clientsecert, 'Content-Type': content_type, 'clientId': clientid, }
    if netsessionid:
        querystring['netsessionid'] = netsessionid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rosacrypto1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_8_getcandles(clientsecert: str, content_type: str, clientid: str, netsessionid: str='c2fEErYaPUAKfOGE4zD15Mh2Nh0=', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://rosacrypto1.p.rapidapi.com/data/oneminutecandle/searchAll"
    querystring = {'clientSecert': clientsecert, 'Content-Type': content_type, 'clientId': clientid, }
    if netsessionid:
        querystring['netsessionid'] = netsessionid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rosacrypto1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_6_getallorders(clientid: str, clientsecert: str, content_type: str, netsessionid: str='c2fEErYaPUAKfOGE4zD15Mh2Nh0=', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://rosacrypto1.p.rapidapi.com/data/order/searchAll"
    querystring = {'clientId': clientid, 'clientSecert': clientsecert, 'Content-Type': content_type, }
    if netsessionid:
        querystring['netsessionid'] = netsessionid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rosacrypto1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_5_getorderbookbuy(clientid: str, clientsecert: str, content_type: str, netsessionid: str='c2fEErYaPUAKfOGE4zD15Mh2Nh0=', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://rosacrypto1.p.rapidapi.com/data/orderBookBuy/"
    querystring = {'clientId': clientid, 'clientsecert': clientsecert, 'Content-Type': content_type, }
    if netsessionid:
        querystring['netsessionid'] = netsessionid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rosacrypto1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_5_getorderbooksell(clientsecert: str, clientid: str, content_type: str, netsessionid: str='c2fEErYaPUAKfOGE4zD15Mh2Nh0=', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://rosacrypto1.p.rapidapi.com/data/orderBookSell/"
    querystring = {'clientSecert': clientsecert, 'clientId': clientid, 'Content-Type': content_type, }
    if netsessionid:
        querystring['netsessionid'] = netsessionid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rosacrypto1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_7_getallmatchedpair_trades(clientsecert: str, content_type: str, clientid: str, netsessionid: str='c2fEErYaPUAKfOGE4zD15Mh2Nh0=', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://rosacrypto1.p.rapidapi.com/data/matchedPair/searchAll"
    querystring = {'clientSecert': clientsecert, 'Content-Type': content_type, 'clientId': clientid, }
    if netsessionid:
        querystring['netsessionid'] = netsessionid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rosacrypto1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_1_getcurrentusersession(clientid: str, content_type: str, clientsecert: str, netsessionid: str='c2fEErYaPUAKfOGE4zD15Mh2Nh0=', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "check your user object using clientId , clien secret,"
    
    """
    url = f"https://rosacrypto1.p.rapidapi.com/data/session/searchall/"
    querystring = {'clientId': clientid, 'Content-Type': content_type, 'clientSecert': clientsecert, }
    if netsessionid:
        querystring['netsessionid'] = netsessionid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rosacrypto1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

