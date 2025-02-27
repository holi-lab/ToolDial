import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def convert_hsv(s: int, h: int, v: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint to converts HSV color code to other color code like Hex , RGB, CMYK,HSL"
    
    """
    url = f"https://convexity.p.rapidapi.com/convert-hsv/"
    querystring = {'s': s, 'h': h, 'v': v, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convexity.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hsv_to_hsl(s: int, h: int, v: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Converts  hsv color code to hsv color code"
    
    """
    url = f"https://convexity.p.rapidapi.com/hsv-to-hsl/"
    querystring = {'s': s, 'h': h, 'v': v, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convexity.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hsv_to_cmyk(h: int, v: int, s: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Converts  hsv color code to cmyk color code"
    
    """
    url = f"https://convexity.p.rapidapi.com/hsv-to-cmyk/"
    querystring = {'h': h, 'v': v, 's': s, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convexity.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hsv_to_hex(v: int, h: int, s: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Converts  hsv color code to hex color code"
    
    """
    url = f"https://convexity.p.rapidapi.com/hsv-to-hex/"
    querystring = {'v': v, 'h': h, 's': s, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convexity.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hsv_to_rgb(s: int, h: int, v: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Converts  hsv color code to rgb color code"
    
    """
    url = f"https://convexity.p.rapidapi.com/hsv-to-rgb/"
    querystring = {'s': s, 'h': h, 'v': v, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convexity.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cmyk_to_hsv(m: int, k: int, y: int, c: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Converts  cmyk color code to hsv color code"
    
    """
    url = f"https://convexity.p.rapidapi.com/cmyk-to-hsv/"
    querystring = {'m': m, 'k': k, 'y': y, 'c': c, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convexity.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hsl_to_hsv(h: int, s: int, l: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Converts  hsl color code to hsv color code"
    
    """
    url = f"https://convexity.p.rapidapi.com/hsl-to-hsv/"
    querystring = {'h': h, 's': s, 'l': l, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convexity.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rgb_to_hsv(r: int, g: int, b: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Converts  rgb color code to hsv color code"
    
    """
    url = f"https://convexity.p.rapidapi.com/rgb-to-hsv/"
    querystring = {'r': r, 'g': g, 'b': b, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convexity.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hex_to_hsv(hex: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Converts  hex color code to  hsv color code"
    
    """
    url = f"https://convexity.p.rapidapi.com/hex-to-hsv/"
    querystring = {'hex': hex, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convexity.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cmyk_to_hsl(c: int, m: int, k: int, y: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Converts  cmyk color code to hsl color code"
    
    """
    url = f"https://convexity.p.rapidapi.com/cmyk-to-hsl/"
    querystring = {'c': c, 'm': m, 'k': k, 'y': y, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convexity.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cmyk_to_rgb(m: int, k: int, c: int, y: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Converts  cmyk color code to rgb color code"
    
    """
    url = f"https://convexity.p.rapidapi.com/cmyk-to-rgb/"
    querystring = {'m': m, 'k': k, 'c': c, 'y': y, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convexity.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cmyk_to_hex(y: int, c: int, m: int, k: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Converts  cmyk color code to hex color code"
    
    """
    url = f"https://convexity.p.rapidapi.com/cmyk-to-hex/"
    querystring = {'y': y, 'c': c, 'm': m, 'k': k, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convexity.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_cmyk(c: int, k: int, m: int, y: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint to converts cmyk color code to other color code like RGB , HSL, Hex"
    
    """
    url = f"https://convexity.p.rapidapi.com/convert-cmyk/"
    querystring = {'c': c, 'k': k, 'm': m, 'y': y, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convexity.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hsl_to_cmyk(s: int, h: int, l: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Converts  hsl color code to cmyk color code"
    
    """
    url = f"https://convexity.p.rapidapi.com/hsl-to-cmyk/"
    querystring = {'s': s, 'h': h, 'l': l, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convexity.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hsl_to_hex(s: int, h: int, l: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Converts  hsl color code to hex color code"
    
    """
    url = f"https://convexity.p.rapidapi.com/hsl-to-hex/"
    querystring = {'s': s, 'h': h, 'l': l, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convexity.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hsl_to_rgb(s: int, h: int, l: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Converts  hsl color code to rgb color code"
    
    """
    url = f"https://convexity.p.rapidapi.com/hsl-to-rgb/"
    querystring = {'s': s, 'h': h, 'l': l, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convexity.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_hsl(s: int, h: int, l: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint to converts HSL color code to other color code like Hex , RGB, CMYK"
    
    """
    url = f"https://convexity.p.rapidapi.com/convert-hsl/"
    querystring = {'s': s, 'h': h, 'l': l, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convexity.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rgb_to_cmyk(r: int, g: int, b: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Converts  rgb color code to cmyk color code"
    
    """
    url = f"https://convexity.p.rapidapi.com/rgb-to-cmyk/"
    querystring = {'r': r, 'g': g, 'b': b, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convexity.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rgb_to_hsl(r: int, g: int, b: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Converts  rgb color code to hsl color code"
    
    """
    url = f"https://convexity.p.rapidapi.com/rgb-to-hsl/"
    querystring = {'r': r, 'g': g, 'b': b, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convexity.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rgb_to_hex(b: int, g: int, r: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Converts  rgb color code to hex color code"
    
    """
    url = f"https://convexity.p.rapidapi.com/rgb-to-hex/"
    querystring = {'b': b, 'g': g, 'r': r, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convexity.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_rgb(r: int, g: int, b: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint to converts RGB color code to other color code like Hex , HSL, CMYK"
    
    """
    url = f"https://convexity.p.rapidapi.com/convert-rgb/"
    querystring = {'r': r, 'g': g, 'b': b, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convexity.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_hex(hex: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint to converts hex color code to other color code like RGB , HSL, CMYK"
    
    """
    url = f"https://convexity.p.rapidapi.com/convert-hex/"
    querystring = {'hex': hex, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convexity.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hex_to_rgb(hex: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Converts hex color code to rgb color code."
    
    """
    url = f"https://convexity.p.rapidapi.com/hex-to-rgb/"
    querystring = {'hex': hex, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convexity.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hex_to_cmyk(hex: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Converts hex color code to  cmyk color code"
    
    """
    url = f"https://convexity.p.rapidapi.com/hex-to-cmyk/"
    querystring = {'hex': hex, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convexity.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hex_to_hsl(hex: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Converts  hex color code to  hsl color code"
    
    """
    url = f"https://convexity.p.rapidapi.com/hex-to-hsl/"
    querystring = {'hex': hex, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convexity.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

