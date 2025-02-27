import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def numbers_base_octal(number: int, is_from: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert a given number to octal"
    number: Number to convert to octal
        is_from: Base of the supplied number (Optional base 10 assumed by default)
        
    """
    url = f"https://numbers7.p.rapidapi.com/numbers/base/octal"
    querystring = {'number': number, }
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "numbers7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def numbers_currency(number: int=None, language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Spells out the number as a currency"
    number: Number to spell
        language: Language to use
        
    """
    url = f"https://numbers7.p.rapidapi.com/numbers/currency"
    querystring = {}
    if number:
        querystring['number'] = number
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "numbers7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def numbers_ordinal(number: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the ordinal of the given number"
    number: Number value
        
    """
    url = f"https://numbers7.p.rapidapi.com/numbers/ordinal"
    querystring = {}
    if number:
        querystring['number'] = number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "numbers7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def numbers_cardinal(number: int=None, language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the cardinal of the given number"
    number: Number value
        language: Language to use
        
    """
    url = f"https://numbers7.p.rapidapi.com/numbers/cardinal"
    querystring = {}
    if number:
        querystring['number'] = number
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "numbers7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def numbers_prime_is_prime(number: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Checks whether a given number is a known prime number or not."
    number: Number to check
        
    """
    url = f"https://numbers7.p.rapidapi.com/numbers/prime/is-prime"
    querystring = {}
    if number:
        querystring['number'] = number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "numbers7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def numbers_pi(is_from: int=None, to: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get digits of pi (Ï€)"
    is_from: Optional start position
        to: Optional start position
        
    """
    url = f"https://numbers7.p.rapidapi.com/numbers/pi"
    querystring = {}
    if is_from:
        querystring['from'] = is_from
    if to:
        querystring['to'] = to
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "numbers7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def numbers_base_hex(number: int, is_from: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert a given number to hexadecimal"
    number: Number to convert to hex
        is_from: Base of the supplied number (Optional base 10 assumed by default)
        
    """
    url = f"https://numbers7.p.rapidapi.com/numbers/base/hex"
    querystring = {'number': number, }
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "numbers7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def numbers_random(total: int=None, max: int=None, min: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate random number(s)"
    total: Total number of random numbers to generate. Defaults to 1
        max: Maximum Number value
        min: Minimum Number value in the range
        
    """
    url = f"https://numbers7.p.rapidapi.com/numbers/random"
    querystring = {}
    if total:
        querystring['total'] = total
    if max:
        querystring['max'] = max
    if min:
        querystring['min'] = min
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "numbers7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def numbers_prime_is_mersenne_prime(number: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Checks whether a given number is a known mersenne prime number or not."
    number: Number to check
        
    """
    url = f"https://numbers7.p.rapidapi.com/numbers/prime/is-mersenne-prime"
    querystring = {}
    if number:
        querystring['number'] = number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "numbers7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def numbers_prime_is_pell_prime(number: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Checks whether a given number is a known pell prime number or not."
    number: Number to check
        
    """
    url = f"https://numbers7.p.rapidapi.com/numbers/prime/is-pell-prime"
    querystring = {}
    if number:
        querystring['number'] = number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "numbers7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def numbers_fact(number: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a random fact about a number"
    number: Number value
        
    """
    url = f"https://numbers7.p.rapidapi.com/numbers/fact"
    querystring = {'number': number, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "numbers7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def numbers_nod(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the number of the day for current day"
    
    """
    url = f"https://numbers7.p.rapidapi.com/numbers/nod"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "numbers7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def numbers_prime_factors(number: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the prime factors of a given number."
    number: Number to get the factors
        
    """
    url = f"https://numbers7.p.rapidapi.com/numbers/prime/factors"
    querystring = {}
    if number:
        querystring['number'] = number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "numbers7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def numbers_numeral_chinese(number: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert base 10 representation of a given number to chinese numeral."
    number: Number to convert
        
    """
    url = f"https://numbers7.p.rapidapi.com/numbers/numeral/chinese"
    querystring = {}
    if number:
        querystring['number'] = number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "numbers7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def numbers_base_binary(number: int, is_from: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert a given number to binary"
    number: Number to convert to binary
        is_from: Base of the supplied number (Optional base 10 assumed by default)
        
    """
    url = f"https://numbers7.p.rapidapi.com/numbers/base/binary"
    querystring = {'number': number, }
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "numbers7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def numbers_prime_is_partition_prime(number: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Checks whether a given number is a known partition prime number or not."
    number: Number to check
        
    """
    url = f"https://numbers7.p.rapidapi.com/numbers/prime/is-partition-prime"
    querystring = {}
    if number:
        querystring['number'] = number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "numbers7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def numbers_numeral_roman(number: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert base 10 representation of a given number to roman numeral."
    number: Number to convert
        
    """
    url = f"https://numbers7.p.rapidapi.com/numbers/numeral/roman"
    querystring = {}
    if number:
        querystring['number'] = number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "numbers7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def numbers_numeral_egyptian(number: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert base 10 representation of a given number to egyptian numeral."
    number: Number to convert
        
    """
    url = f"https://numbers7.p.rapidapi.com/numbers/numeral/egyptian"
    querystring = {}
    if number:
        querystring['number'] = number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "numbers7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def numbers_prime_is_fibonacci_prime(number: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Checks whether a given number is a known fibonacci prime number or not."
    number: Number to check
        
    """
    url = f"https://numbers7.p.rapidapi.com/numbers/prime/is-fibonacci-prime"
    querystring = {}
    if number:
        querystring['number'] = number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "numbers7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def numbers_base(to: int, number: int, is_from: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert a given number from one base to another base"
    to: Target base to convert to
        number: Number to convert to the target base
        is_from: Base of the supplied number (Optional base 10 assumed by default)
        
    """
    url = f"https://numbers7.p.rapidapi.com/numbers/base"
    querystring = {'to': to, 'number': number, }
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "numbers7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def numbers_is_palindrome(number: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Checks whether a given number is a palindrome number or not."
    number: Number to check
        
    """
    url = f"https://numbers7.p.rapidapi.com/numbers/is-palindrome"
    querystring = {}
    if number:
        querystring['number'] = number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "numbers7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def numbers_prime_is_fermat_prime(number: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Checks whether a given number is a known fermat prime number or not."
    number: Number to check
        
    """
    url = f"https://numbers7.p.rapidapi.com/numbers/prime/is-fermat-prime"
    querystring = {}
    if number:
        querystring['number'] = number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "numbers7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def numbers_is_cube(number: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Checks whether a given number is a cube number or not."
    number: Number to check
        
    """
    url = f"https://numbers7.p.rapidapi.com/numbers/is-cube"
    querystring = {}
    if number:
        querystring['number'] = number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "numbers7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def numbers_is_square(number: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Checks whether a given number is a square number or not."
    number: Number to check
        
    """
    url = f"https://numbers7.p.rapidapi.com/numbers/is-square"
    querystring = {}
    if number:
        querystring['number'] = number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "numbers7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def numbers_is_triangle(number: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Checks whether a given number is a triangle number or not."
    number: Number to check
        
    """
    url = f"https://numbers7.p.rapidapi.com/numbers/is-triangle"
    querystring = {}
    if number:
        querystring['number'] = number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "numbers7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def numbers_prime_is_perfect(number: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Checks whether a given number is a perfect number or not."
    number: Number to check
        
    """
    url = f"https://numbers7.p.rapidapi.com/numbers/prime/is-perfect"
    querystring = {}
    if number:
        querystring['number'] = number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "numbers7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

