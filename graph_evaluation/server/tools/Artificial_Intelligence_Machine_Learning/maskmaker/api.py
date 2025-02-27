import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def grid(height: int, width: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Random grids: Generate random grids of transparent squares or rectangles with varying sizes and transparency.
		height and width are required and must be a value between 1 - 10000"
    
    """
    url = f"https://maskmaker.p.rapidapi.com/grid"
    querystring = {'height': height, 'width': width, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maskmaker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cellular_automata(width: int, height: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Cellular automata: Use cellular automata to generate a pattern of transparent and opaque cells that evolve over time.
		height and width are required and must be a value between 1 - 10000"
    
    """
    url = f"https://maskmaker.p.rapidapi.com/cellularautomata"
    querystring = {'width': width, 'height': height, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maskmaker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def concentric_rings(width: int=256, height: int=256, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generates concentric rings
		height and width are required and must be a value between 1 - 10000"
    
    """
    url = f"https://maskmaker.p.rapidapi.com/conring"
    querystring = {}
    if width:
        querystring['width'] = width
    if height:
        querystring['height'] = height
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maskmaker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def scattered_rings(height: int, width: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generates scattered rings
		height and width are required and must be a value between 1 - 10000"
    
    """
    url = f"https://maskmaker.p.rapidapi.com/ring"
    querystring = {'height': height, 'width': width, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maskmaker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def circular(width: int, height: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Produces sections of varying transparency in circular sections
		
		height and width are required and must be a value between 1 - 10000"
    
    """
    url = f"https://maskmaker.p.rapidapi.com/circular"
    querystring = {'width': width, 'height': height, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maskmaker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hightrans(width: int, height: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "height and width are required and must be a value between 1 - 10000"
    
    """
    url = f"https://maskmaker.p.rapidapi.com/hightrans"
    querystring = {'width': width, 'height': height, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maskmaker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def spiral(width: int, height: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Produces sections of varying transparency in circular sections
		
		height and width are required and must be a value between 1 - 10000"
    
    """
    url = f"https://maskmaker.p.rapidapi.com/spiral"
    querystring = {'width': width, 'height': height, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maskmaker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_mask(width: int, height: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Produces sections of varying transparency in grid-like pattern of rectangular sections
		
		height and width are required and must be a value between 1 - 10000"
    
    """
    url = f"https://maskmaker.p.rapidapi.com/random"
    querystring = {'width': width, 'height': height, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maskmaker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_pixel(height: int, width: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "height and width are required and must be a value between 1 - 10000"
    
    """
    url = f"https://maskmaker.p.rapidapi.com/randompixel"
    querystring = {'height': height, 'width': width, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maskmaker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_diamond(height: int, width: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Produces sections of varying transparency in grid-like pattern of rectangular sections
		
		height and width are required and must be a value between 1 - 10000"
    
    """
    url = f"https://maskmaker.p.rapidapi.com/diamond"
    querystring = {'height': height, 'width': width, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maskmaker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

