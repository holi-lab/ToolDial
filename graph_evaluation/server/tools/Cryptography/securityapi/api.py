import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def base64_encrypt(data: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "base64 encrypt
		JSON: {"msg": message}"
    
    """
    url = f"https://securityapi1.p.rapidapi.com/base64/encrypt/{data}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "securityapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def base64_decrypt(data: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "base64 decrypt
		JSON:{"msg": base64_encoded_message}"
    
    """
    url = f"https://securityapi1.p.rapidapi.com/base64/decrypt/{data}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "securityapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def password_checker(json: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "check if password encripted 
		Algo Suported:
		MD4
		MD5
		SHA-1
		SHA-224
		SHA-256
		SHA-384
		SHA-512
		SHA3-224
		SHA3-256
		SHA3-384
		SHA3-512
		RIPEMD-160
		RIPEMD-320
		HAS-160
		Whirlpool
		JSON: {"algo":"sha256","pass":hashed_password,"enter_pass":user password}"
    
    """
    url = f"https://securityapi1.p.rapidapi.com/pass/{json}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "securityapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hmac(json: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "verify a hmac
		JSON: {"algo": "hmac", "pass":hashed_password,"enter_pass":user password, "key": the_secret_key}"
    
    """
    url = f"https://securityapi1.p.rapidapi.com/pass/{json}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "securityapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "test endpoint"
    
    """
    url = f"https://securityapi1.p.rapidapi.com/api"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "securityapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_rsa_keys(format: str='format=true', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get rsa keys"
    format: the output with be with out the \\\\n special chars
        
    """
    url = f"https://securityapi1.p.rapidapi.com/rsa/key/{format}"
    querystring = {}
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "securityapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def decrypt_with_aes(json: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "decrypt with aes 
		json: {"key": aes_key, "iv": aes_iv, "enc": encrypted_message}
		aes suported aes-256-cbc"
    
    """
    url = f"https://securityapi1.p.rapidapi.com/aes/decrypt/{json}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "securityapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def encrypt_with_aes(json: str, data: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "encrypt with aes
		json: {"key": aes_key, "iv": aes_iv, "msg": message}
		aes suported aes-256-cbc"
    
    """
    url = f"https://securityapi1.p.rapidapi.com/aes/encrypt/{json}"
    querystring = {}
    if data:
        querystring['data'] = data
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "securityapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_random_aes_key(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get JSON with random aes and iv key"
    
    """
    url = f"https://securityapi1.p.rapidapi.com/aes/key"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "securityapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

