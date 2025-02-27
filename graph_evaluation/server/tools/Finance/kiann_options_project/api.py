import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def opt_rho(type: str, ret: int, r: int, k: int, sigma: int, s: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "option lognormal greeks : delta calculates the delta, given where
		S = forward (float), K = strike (float), T = time-to-expiry (float), sigma = implied volatility lognormal (float)
		r = risk-free-rate (float), type_  =call (c) /put (p) (string), ret_= 0 (default) for use in rapidAPI.com"
    
    """
    url = f"https://kiann_options_project.p.rapidapi.com/opt_rho"
    querystring = {'type_': type, 'ret_': ret, 'r': r, 'K': k, 'sigma': sigma, 'S': s, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kiann_options_project.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def opt_theta(r: int, ret: int, type: str, sigma: int, k: int, s: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "option lognormal greeks : theta calculates the theta, given where
		S = forward (float), K = strike (float), T = time-to-expiry (float), sigma = implied volatility lognormal (float)
		r = risk-free-rate (float), type_  =call (c) /put (p) (string), ret_= 0 (default) for use in rapidAPI.com"
    
    """
    url = f"https://kiann_options_project.p.rapidapi.com/opt_theta"
    querystring = {'r': r, 'ret_': ret, 'type_': type, 'sigma': sigma, 'K': k, 'S': s, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kiann_options_project.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def opt_vega(ret: int, r: int, k: int, s: int, type: str, sigma: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "option lognormal greeks : vega calculates the vega, given where
		S = forward (float), K = strike (float), T = time-to-expiry (float), sigma = implied volatility lognormal (float)
		r = risk-free-rate (float), type_  =call (c) /put (p) (string), ret_ = 0 (default) for use in rapidAPI.com"
    
    """
    url = f"https://kiann_options_project.p.rapidapi.com/opt_vega"
    querystring = {'ret_': ret, 'r': r, 'K': k, 'S': s, 'type_': type, 'sigma': sigma, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kiann_options_project.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def opt_gamma(ret: int, sigma: int, r: int, k: int, type: str, s: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "option lognormal greeks : gamma calculates the gamma, given where
		S = forward (float), K = strike (float), T = time-to-expiry (float), sigma = implied volatility lognormal (float)
		r = risk-free-rate (float), cp  =call (c) /put (p) (string), type_ = 0 (default) for use in rapidAPI.com"
    
    """
    url = f"https://kiann_options_project.p.rapidapi.com/opt_gamma"
    querystring = {'ret_': ret, 'sigma': sigma, 'r': r, 'K': k, 'type_': type, 'S': s, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kiann_options_project.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def opt_delta(type: str, t: int, sigma: int, ret: int, s: int, k: int, r: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "option lognormal greeks : delta calculates the delta, given where
		S = forward (float), K = strike (float), T = time-to-expiry (float), sigma = implied volatility lognormal (float)
		r = risk-free-rate (float), cp  =call (c) /put (p) (string), type_ = 0 (default) for use in rapidAPI.com
		
		
		r: float, S: float, K: float, T: float, sigma: float, type_: str ="c", ret_: int = 1"
    
    """
    url = f"https://kiann_options_project.p.rapidapi.com/opt_delta"
    querystring = {'type_': type, 'T': t, 'sigma': sigma, 'ret_': ret, 'S': s, 'K': k, 'r': r, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kiann_options_project.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def normal_call(ret: str, f: int, r: int, cp: str, v: int, k: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "params = {"k": 1.0, "f": 1.0, "t": 1.0, "v":0.3, "r":0.0, "cp":"call"}"
    
    """
    url = f"https://kiann_options_project.p.rapidapi.com/normal_call"
    querystring = {'ret_': ret, 'f': f, 'r': r, 'cp': cp, 'v': v, 'k': k, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kiann_options_project.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lognormal_call(ret: int, v: int, r: int, cp: str, k: int, f: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Black-Scholes Pricer/premium, Lognormal vols. Params are as per :
		k = strike (float)
		f = forward (float)
		t = time-to-expiry (float)
		v = implied volatility, lognormal, annualized (float)
		r = risk-free-rate (float). Note can be combined with dividend, funding, risk-free rate into one value"
    
    """
    url = f"https://kiann_options_project.p.rapidapi.com/lognormal_call"
    querystring = {'ret_': ret, 'v': v, 'r': r, 'cp': cp, 'k': k, 'f': f, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kiann_options_project.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sunction_one(x2: int, x1: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "N/A"
    
    """
    url = f"https://kiann_options_project.p.rapidapi.com/Sunction-one"
    querystring = {'x2': x2, 'x1': x1, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kiann_options_project.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

