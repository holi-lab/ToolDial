import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getmerdeka(q: str='covid', lim: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get from Merdeka News"
    
    """
    url = f"https://indonews.p.rapidapi.com/merdeka"
    querystring = {}
    if q:
        querystring['q'] = q
    if lim:
        querystring['lim'] = lim
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indonews.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmediaindonesia(q: str='covid', lim: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get from MediaIndonesia News"
    
    """
    url = f"https://indonews.p.rapidapi.com/mediaindonesia"
    querystring = {}
    if q:
        querystring['q'] = q
    if lim:
        querystring['lim'] = lim
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indonews.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettribun(lim: int=20, q: str='covid', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get from Tribun News"
    
    """
    url = f"https://indonews.p.rapidapi.com/tribun"
    querystring = {}
    if lim:
        querystring['lim'] = lim
    if q:
        querystring['q'] = q
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indonews.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsuara(q: str='covid', lim: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get from Suara News"
    
    """
    url = f"https://indonews.p.rapidapi.com/suara"
    querystring = {}
    if q:
        querystring['q'] = q
    if lim:
        querystring['lim'] = lim
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indonews.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsindonews(q: str='covid', lim: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get from SindoNews"
    
    """
    url = f"https://indonews.p.rapidapi.com/sindo"
    querystring = {}
    if q:
        querystring['q'] = q
    if lim:
        querystring['lim'] = lim
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indonews.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getliputan6(q: str='covid', lim: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get from Liputan6 News"
    
    """
    url = f"https://indonews.p.rapidapi.com/liputan6"
    querystring = {}
    if q:
        querystring['q'] = q
    if lim:
        querystring['lim'] = lim
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indonews.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getjpnn(lim: int=20, q: str='covid', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get from JPNN News"
    
    """
    url = f"https://indonews.p.rapidapi.com/jpnn"
    querystring = {}
    if lim:
        querystring['lim'] = lim
    if q:
        querystring['q'] = q
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indonews.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdetik(q: str='covid', lim: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get from Detik News"
    
    """
    url = f"https://indonews.p.rapidapi.com/detik"
    querystring = {}
    if q:
        querystring['q'] = q
    if lim:
        querystring['lim'] = lim
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indonews.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getindex(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get All News from available news provider"
    
    """
    url = f"https://indonews.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indonews.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettempo(lim: int=20, q: str='covid', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get from Tempo News"
    
    """
    url = f"https://indonews.p.rapidapi.com/tempo"
    querystring = {}
    if lim:
        querystring['lim'] = lim
    if q:
        querystring['q'] = q
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indonews.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getkompas(q: str='covid', lim: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get from Kompas News"
    
    """
    url = f"https://indonews.p.rapidapi.com/kompas"
    querystring = {}
    if q:
        querystring['q'] = q
    if lim:
        querystring['lim'] = lim
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indonews.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

