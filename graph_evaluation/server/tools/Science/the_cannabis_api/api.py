import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_all_effects(accept: str='text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Every Cannabis Effect in the dataset."
    
    """
    url = f"https://the-cannabis-api.p.rapidapi.com/strains/getAllEffects"
    querystring = {}
    if accept:
        querystring['Accept'] = accept
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-cannabis-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_flavours(accept: str='text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Every Cannabis Flavour in the dataset."
    
    """
    url = f"https://the-cannabis-api.p.rapidapi.com/strains/getAllFlavors"
    querystring = {}
    if accept:
        querystring['Accept'] = accept
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-cannabis-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_strain_flavours_by_id(strainid: str, accept: str='text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the flavours of the strain with the specified ID."
    
    """
    url = f"https://the-cannabis-api.p.rapidapi.com/strains/getFlavorsByStrainId/{strainid}"
    querystring = {}
    if accept:
        querystring['Accept'] = accept
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-cannabis-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_strain_effects_by_id(strainid: str, accept: str='text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the effects of the strain with the specified ID."
    
    """
    url = f"https://the-cannabis-api.p.rapidapi.com/strains/getEffectsByStrainId/{strainid}"
    querystring = {}
    if accept:
        querystring['Accept'] = accept
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-cannabis-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_strain_by_flavours(strainflavour: str, accept: str='text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the id, name, type, rating, effects, flavours and description of all the strains with the specified flavours."
    
    """
    url = f"https://the-cannabis-api.p.rapidapi.com/strains/getStrainsByFlavour/{strainflavour}"
    querystring = {}
    if accept:
        querystring['Accept'] = accept
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-cannabis-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_strain_by_effects(straineffect: str, accept: str='text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the id, name, type, rating, effects, flavours and description of all the strains with the specified effects."
    
    """
    url = f"https://the-cannabis-api.p.rapidapi.com/strains/getStrainsByEffect/{straineffect}"
    querystring = {}
    if accept:
        querystring['Accept'] = accept
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-cannabis-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_strain_by_type(straintype: str, accept: str='text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the id, name, type, rating, effects, flavours and description of all the strains with the specified type (sativa, indica or hybrid)."
    
    """
    url = f"https://the-cannabis-api.p.rapidapi.com/strains/getStrainsByType/{straintype}"
    querystring = {}
    if accept:
        querystring['Accept'] = accept
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-cannabis-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_strain_by_name(strainname: str, accept: str='text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the id, name, type, rating, flavours, effects and description of a strain by it's name."
    
    """
    url = f"https://the-cannabis-api.p.rapidapi.com/strains/getStrainsByName/{strainname}"
    querystring = {}
    if accept:
        querystring['Accept'] = accept
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-cannabis-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

