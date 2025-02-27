import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def companies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of all available companies that can be used in /laptop/search ?company filter"
    
    """
    url = f"https://laptopdb1.p.rapidapi.com/companies"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "laptopdb1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gpu_search(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lets you search gpus by name"
    
    """
    url = f"https://laptopdb1.p.rapidapi.com/gpu/search"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "laptopdb1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cpu_search(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lets you search cpus by name"
    
    """
    url = f"https://laptopdb1.p.rapidapi.com/cpu/search"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "laptopdb1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def laptop_search(security: str=None, company: str=None, inches: str=None, q: str=None, sort: str=None, cpu: str=None, gpu: int=None, ram: str=None, os: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lets you search laptops by specs or name"
    inches: Example: 10.1,11.6,12, etc.. You can also send multiple comma separated.
        cpu: It can be simple like i3 or i9, but also an specific integer.
        gpu: GPU id that can be obtained using gpu/search call. Ex: 11 (Nvidia 1080 GTX)
        
    """
    url = f"https://laptopdb1.p.rapidapi.com/laptop/search"
    querystring = {}
    if security:
        querystring['security'] = security
    if company:
        querystring['company'] = company
    if inches:
        querystring['inches'] = inches
    if q:
        querystring['q'] = q
    if sort:
        querystring['sort'] = sort
    if cpu:
        querystring['cpu'] = cpu
    if gpu:
        querystring['gpu'] = gpu
    if ram:
        querystring['ram'] = ram
    if os:
        querystring['os'] = os
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "laptopdb1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

