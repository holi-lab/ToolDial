import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_brand_by_id(brandid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Grab a Brand by it's Database ID"
    
    """
    url = f"https://cigars.p.rapidapi.com/brands/{brandid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cigars.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_brands(page: int, search: str='Dominican', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Cigar brands, paginated and searchable."
    
    """
    url = f"https://cigars.p.rapidapi.com/brands"
    querystring = {'page': page, }
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cigars.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_cigar_strengths(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Enum of cigar strengths"
    
    """
    url = f"https://cigars.p.rapidapi.com/cigars/strengths"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cigars.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_cigar_average_ring_gauge(brandid: int=13711, filler: str='Nicaragua', wrapper: str='Connecticut Shade, Ecuador', country: str='Nicaragua', color: str='Mild-Medium', strength: str='Colorado Claro', name: str='Maduro', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the average ring gauge of cigars, filter supported"
    
    """
    url = f"https://cigars.p.rapidapi.com/cigars/averageRingGauge"
    querystring = {}
    if brandid:
        querystring['brandId'] = brandid
    if filler:
        querystring['filler'] = filler
    if wrapper:
        querystring['wrapper'] = wrapper
    if country:
        querystring['country'] = country
    if color:
        querystring['color'] = color
    if strength:
        querystring['strength'] = strength
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cigars.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_cigar_average_length(wrapper: str='Connecticut Shade, Ecuador', name: str='Torpedo', filler: str='Nicaragua', country: str='Nicaragua', color: str='Colorado Claro', strength: str='Mild-Medium', brandid: int=13711, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the average length of cigars, filter supported"
    
    """
    url = f"https://cigars.p.rapidapi.com/cigars/averageLength"
    querystring = {}
    if wrapper:
        querystring['wrapper'] = wrapper
    if name:
        querystring['name'] = name
    if filler:
        querystring['filler'] = filler
    if country:
        querystring['country'] = country
    if color:
        querystring['color'] = color
    if strength:
        querystring['strength'] = strength
    if brandid:
        querystring['brandId'] = brandid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cigars.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_cigar_by_id(cigarid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a Cigar by it's Database ID"
    
    """
    url = f"https://cigars.p.rapidapi.com/cigars/{cigarid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cigars.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_colors(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Enum of Cigar colors"
    
    """
    url = f"https://cigars.p.rapidapi.com/cigars/colors"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cigars.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_cigars(page: int, color: str='Claro', filler: str='Nicaragua', country: str='Nicaragua', strength: str='Mild', wrapper: str='Connecticut', name: str='Connecticut', brandid: int=13711, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get cigar data, paginated with filters"
    name: Uses similar search to find a Cigar by name
        
    """
    url = f"https://cigars.p.rapidapi.com/cigars"
    querystring = {'page': page, }
    if color:
        querystring['color'] = color
    if filler:
        querystring['filler'] = filler
    if country:
        querystring['country'] = country
    if strength:
        querystring['strength'] = strength
    if wrapper:
        querystring['wrapper'] = wrapper
    if name:
        querystring['name'] = name
    if brandid:
        querystring['brandId'] = brandid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cigars.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

