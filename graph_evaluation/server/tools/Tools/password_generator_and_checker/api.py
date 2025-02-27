import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_strong_password_strong(symbols: bool=None, excludesimilarcharacters: bool=None, lowercase: bool=None, uppercase: bool=None, length: int=10, numbers: bool=None, exclude: str='+`', strict: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate strong passwords. Can be called without any parameters or with parameters for further customization."
    strict: teste
        
    """
    url = f"https://password-generator-and-checker.p.rapidapi.com/generator/strong"
    querystring = {}
    if symbols:
        querystring['symbols'] = symbols
    if excludesimilarcharacters:
        querystring['excludeSimilarCharacters'] = excludesimilarcharacters
    if lowercase:
        querystring['lowercase'] = lowercase
    if uppercase:
        querystring['uppercase'] = uppercase
    if length:
        querystring['length'] = length
    if numbers:
        querystring['numbers'] = numbers
    if exclude:
        querystring['exclude'] = exclude
    if strict:
        querystring['strict'] = strict
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "password-generator-and-checker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generate_very_strong_password_verystrong(strict: bool=None, uppercase: bool=None, lowercase: bool=None, numbers: bool=None, excludesimilarcharacters: bool=None, length: int=16, exclude: str=None, symbols: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate very strong passwords. Can be called without any parameters or with parameters for further customization."
    
    """
    url = f"https://password-generator-and-checker.p.rapidapi.com/generator/verystrong"
    querystring = {}
    if strict:
        querystring['strict'] = strict
    if uppercase:
        querystring['uppercase'] = uppercase
    if lowercase:
        querystring['lowercase'] = lowercase
    if numbers:
        querystring['numbers'] = numbers
    if excludesimilarcharacters:
        querystring['excludeSimilarCharacters'] = excludesimilarcharacters
    if length:
        querystring['length'] = length
    if exclude:
        querystring['exclude'] = exclude
    if symbols:
        querystring['symbols'] = symbols
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "password-generator-and-checker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def average_password_average(strict: bool=None, exclude: str='p', length: int=8, uppercase: bool=None, symbols: bool=None, excludesimilarcharacters: bool=None, lowercase: bool=None, numbers: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a password of medium complexity. Allows for further customization."
    
    """
    url = f"https://password-generator-and-checker.p.rapidapi.com/generator/average"
    querystring = {}
    if strict:
        querystring['strict'] = strict
    if exclude:
        querystring['exclude'] = exclude
    if length:
        querystring['length'] = length
    if uppercase:
        querystring['uppercase'] = uppercase
    if symbols:
        querystring['symbols'] = symbols
    if excludesimilarcharacters:
        querystring['excludeSimilarCharacters'] = excludesimilarcharacters
    if lowercase:
        querystring['lowercase'] = lowercase
    if numbers:
        querystring['numbers'] = numbers
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "password-generator-and-checker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def captcha(value: str='Sample text', length: int=11, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate random captchas or provide a value for a specific captcha generation.
		Optional values are:
		- length: generate a captcha with a random word of a given length
		- value: generate a captcha for the provided word."
    
    """
    url = f"https://password-generator-and-checker.p.rapidapi.com/captcha/"
    querystring = {}
    if value:
        querystring['value'] = value
    if length:
        querystring['length'] = length
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "password-generator-and-checker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def password_strength_checker(password: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a integer from 0-4 (useful for implementing a strength bar):
		- 0 # too guessable: risky password. (guesses < 10^3)
		- 1 # very guessable: protection from throttled online attacks. (guesses < 10^6)
		- 2 # somewhat guessable: protection from unthrottled online attacks. (guesses < 10^8)
		- 3 # safely unguessable: moderate protection from offline slow-hash scenario. (guesses < 10^10)
		- 4 # very unguessable: strong protection from offline slow-hash scenario. (guesses >= 10^10)
		
		Also returns other useful information, such as the estimated time to crack"
    
    """
    url = f"https://password-generator-and-checker.p.rapidapi.com/checker/{password}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "password-generator-and-checker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generate_custom_password_password(length: int=12, numbers: bool=None, excludesimilarcharacters: bool=None, uppercase: bool=None, exclude: str='p', lowercase: bool=None, symbols: bool=None, strict: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate strong passwords. Can be called without any parameters and generates a password with 10 characters or with parameters for further customization."
    
    """
    url = f"https://password-generator-and-checker.p.rapidapi.com/generator/password"
    querystring = {}
    if length:
        querystring['length'] = length
    if numbers:
        querystring['numbers'] = numbers
    if excludesimilarcharacters:
        querystring['excludeSimilarCharacters'] = excludesimilarcharacters
    if uppercase:
        querystring['uppercase'] = uppercase
    if exclude:
        querystring['exclude'] = exclude
    if lowercase:
        querystring['lowercase'] = lowercase
    if symbols:
        querystring['symbols'] = symbols
    if strict:
        querystring['strict'] = strict
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "password-generator-and-checker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

