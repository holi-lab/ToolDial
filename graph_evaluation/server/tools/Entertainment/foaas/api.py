import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def fuck_off(name: str, is_from: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Will return content of the form 'Fuck off, :name. - :from', e.g. /off/Tom/Chris will return 'Fuck off, Tom - Chris'"
    
    """
    url = f"https://mashape-community-foaas.p.rapidapi.com/off/{name}/{is_from}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashape-community-foaas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fuck_my_life(is_from: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Will return content of the form 'Well, Fuck me pink. - :from'."
    
    """
    url = f"https://mashape-community-foaas.p.rapidapi.com/life/{is_from}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashape-community-foaas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fuck_that(is_from: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Will return content of the form 'Fuck that. - :from', e.g. /that/Chris will return 'Fuck that. - Chris'"
    
    """
    url = f"https://mashape-community-foaas.p.rapidapi.com/that/{is_from}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashape-community-foaas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fuck_you_very_much(is_from: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Will return content of the form 'Fuck you very much. - :from'."
    
    """
    url = f"https://mashape-community-foaas.p.rapidapi.com/thanks/{is_from}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashape-community-foaas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fuck_you(name: str, is_from: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Will return content of the form 'Fuck you, :name. - :from', e.g. /you/Tom/Chris will return 'Fuck you, Tom - Chris'"
    
    """
    url = f"https://mashape-community-foaas.p.rapidapi.com/you/{name}/{is_from}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashape-community-foaas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def king(name: str, is_from: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Will return content of the form 'Oh fuck off, just really fuck off you total dickface. Christ :name, you are fucking thick. - :from'."
    
    """
    url = f"https://mashape-community-foaas.p.rapidapi.com/king/{name}/{is_from}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashape-community-foaas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def chainsaw(name: str, is_from: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Will return content of the form 'Fuck me gently with a chainsaw, :name. Do I look like Mother Teresa? - :from', e.g. /chainsaw/Chris/Heather will return 'Fuck me gently with a chainsaw, Chris. Do I look like Mother Teresa? - Heather"
    
    """
    url = f"https://mashape-community-foaas.p.rapidapi.com/chainsaw/{name}/{is_from}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashape-community-foaas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def donut(name: str, is_from: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Will return content of the form ':name, go and take a flying fuck at a rolling donut. - :from', e.g. /donut/Tom/Chris will return 'Tom, go and take a flying fuck at a rolling donut. - Chris'"
    
    """
    url = f"https://mashape-community-foaas.p.rapidapi.com/donut/{name}/{is_from}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashape-community-foaas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fuck_this(is_from: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Will return content of the form 'Fuck this - :from', e.g. /this/Chris will return 'Fuck this. - Chris'"
    
    """
    url = f"https://mashape-community-foaas.p.rapidapi.com/this/{is_from}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashape-community-foaas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fuck_everything(is_from: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Will return content of the form 'Fuck everything. - :from', e.g. /everything/Chris will return 'Fuck everything. - Chris'"
    
    """
    url = f"https://mashape-community-foaas.p.rapidapi.com/everything/{is_from}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashape-community-foaas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shakespeare(name: str, is_from: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Will return content of the form ':name, Thou clay-brained guts, thou knotty-pated fool, thou whoreson obscene greasy tallow-catch! - :from', e.g. /shakespeare/Falstaff/Prince%20Henry will return 'Falstaff, Thou clay-brained guts, thou knotty-pated fool, thou whoreson obscene greasy tallow-catch! - Prince Henry"
    
    """
    url = f"https://mashape-community-foaas.p.rapidapi.com/shakespeare/{name}/{is_from}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashape-community-foaas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fuck_everyone(is_from: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Will return content of the form 'Everyone can go and fuck off. - :name', e.g. /everyone/Tom will return 'Everyone can go and fuck off. - Tom'"
    
    """
    url = f"https://mashape-community-foaas.p.rapidapi.com/everyone/{is_from}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashape-community-foaas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def linus(name: str, is_from: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Will return content of the form ':name, there aren't enough swear-words in the English language, so now I'll have to call you perkeleen vittupää just to express my disgust and frustration with this crap. - :from'."
    
    """
    url = f"https://mashape-community-foaas.p.rapidapi.com/linus/{name}/{is_from}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashape-community-foaas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fuck_me_pink(is_from: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Will return content of the form 'Well, Fuck me pink. - :from'."
    
    """
    url = f"https://mashape-community-foaas.p.rapidapi.com/pink/{is_from}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashape-community-foaas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

