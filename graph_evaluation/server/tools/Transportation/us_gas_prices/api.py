import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def us_hi(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for Hawaii. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/hi"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_id(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for Idaho. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/id"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_in(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for Indiana. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/in"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_ia(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for Iowa. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/ia"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_ks(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for Kansas. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/ks"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_ky(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for Kentucky. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/ky"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_me(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for Maine. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/me"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_md(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for Maryland. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/md"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_mi(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for Michigan. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/mi"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_mn(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for Minnesota. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/mn"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_ms(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for Mississippi. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/ms"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_mo(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for Missouri. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/mo"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_mt(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for Montana. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/mt"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_ne(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for Nebraska. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/ne"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_nj(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for New Jersey. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/nj"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_nm(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for New Mexico. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/nm"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_ny(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for New York. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/ny"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_nc(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for North Carolina. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/nc"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_nd(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for North Dakota. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/nd"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_oh(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for Ohio. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/oh"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_ok(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for Oklahoma. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/ok"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_or(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for Oregon. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/or"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_pa(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for Pennsylvania. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/pa"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_ri(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for Rhode Island. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/ri"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_sc(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for South Carolina. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/sc"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_sd(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for South Dakota. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/sd"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_tn(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for Tennessee. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/tn"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_tx(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for Texas. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/tx"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_ut(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for Utah. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/ut"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_vt(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for Vermont. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/vt"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_va(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for Virginia. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/va"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_wa(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for Washington. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/wa"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_wv(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for West Virginia. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/wv"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_wi(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for Wisconsin. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/wi"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_wy(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for Wyoming. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/wy"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_nh(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for New Hampshire. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/nh"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_nv(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for Nevada. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/nv"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_ma(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for Massachusetts. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/ma"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_la(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for Louisiana. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/la"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_il(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for Illinois. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/il"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_ct(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for Connecticut. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/ct"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_co(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for Colorado. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/co"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_ak(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for Alaska. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/ak"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_az(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for Arizona. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/az"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_de(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for Delaware. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/de"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_dc(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for Washington D.C. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/dc"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_al(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for Alabama. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/al"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_ca(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for California. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/ca"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_fl(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for Florida. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/fl"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_ga(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for Georgia. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/ga"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_ar(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current gas price data for Arkansas. Historical data starting from 2023-06-01 can be queried by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us/ar"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current national average gas price data. You can query historical data starting from 2023-06-01 by adding the `date` parameter to this endpoint. In order to return XML instead of the default JSON format, simply add the `Accept: application/xml` header to your request."
    
    """
    url = f"https://us-gas-prices.p.rapidapi.com/us"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-gas-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

