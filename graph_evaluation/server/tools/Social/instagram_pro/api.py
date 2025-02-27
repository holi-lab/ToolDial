import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def following_by_pk(pk: str, corsenabled: str='false', nextmaxid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Instagram Following List by PK, Up to 1000 records"
    
    """
    url = f"https://instagram-pro.p.rapidapi.com/pro/following"
    querystring = {'pk': pk, }
    if corsenabled:
        querystring['corsEnabled'] = corsenabled
    if nextmaxid:
        querystring['nextMaxId'] = nextmaxid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def following_by_username(username: str, corsenabled: str=None, nextmaxid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Instagram Following List by username, Up to 1000 records"
    
    """
    url = f"https://instagram-pro.p.rapidapi.com/pro/following"
    querystring = {'username': username, }
    if corsenabled:
        querystring['corsEnabled'] = corsenabled
    if nextmaxid:
        querystring['nextMaxId'] = nextmaxid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def followers_by_pk(pk: str, nextmaxid: str=None, corsenabled: str='false', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Instagram Followers list up to 1000 records only"
    
    """
    url = f"https://instagram-pro.p.rapidapi.com/pro/followers"
    querystring = {'pk': pk, }
    if nextmaxid:
        querystring['nextMaxId'] = nextmaxid
    if corsenabled:
        querystring['corsEnabled'] = corsenabled
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def followers_by_username(username: str, nextmaxid: str=None, corsenabled: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Followers list by username, Up to 1000 records"
    
    """
    url = f"https://instagram-pro.p.rapidapi.com/pro/followers"
    querystring = {'username': username, }
    if nextmaxid:
        querystring['nextMaxId'] = nextmaxid
    if corsenabled:
        querystring['corsEnabled'] = corsenabled
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def advance_profile(ig: str, response_type: str, corsenabled: str='false', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Profile data with email/phone if available"
    
    """
    url = f"https://instagram-pro.p.rapidapi.com/advance_profile"
    querystring = {'ig': ig, 'response_type': response_type, }
    if corsenabled:
        querystring['corsEnabled'] = corsenabled
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

