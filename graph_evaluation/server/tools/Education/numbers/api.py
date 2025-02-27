import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_date_fact(month: str, day: str, fragment: str='True', json: str='True', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a fact about a day of year"
    month: The 1-indexed month (eg. 6 for June)
        day: The day of the month
        fragment: Add "?fragment=true" to return the fact as a sentence fragment that can be easily included as part of a larger sentence. This means that the first word is lowercase and ending punctuation is omitted. For trivia and math, a noun phrase is returned that can be used in a sentence like “We now have more users than [fact as fragment]!”.
        json: Specify "true" to return result as JSON instead of plaintext.
        
    """
    url = f"https://numbersapi.p.rapidapi.com/{month}/{day}/date"
    querystring = {}
    if fragment:
        querystring['fragment'] = fragment
    if json:
        querystring['json'] = json
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "numbersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_math_fact(number: str, fragment: str='True', json: str='True', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a mathematical property about a number"
    number: The integer of interest
        fragment: Add "?fragment=true" to return the fact as a sentence fragment that can be easily included as part of a larger sentence. This means that the first word is lowercase and ending punctuation is omitted. For trivia and math, a noun phrase is returned that can be used in a sentence like “We now have more users than [fact as fragment]!”.
        json: Specify "true" to return result as JSON instead of plaintext.
        
    """
    url = f"https://numbersapi.p.rapidapi.com/{number}/math"
    querystring = {}
    if fragment:
        querystring['fragment'] = fragment
    if json:
        querystring['json'] = json
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "numbersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_random_fact(type: str, min: str='10', max: str='20', fragment: str='True', json: str='True', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get result by random number"
    type: One of "trivia", "math", "date", or "year"
        min: Minimum number, inclusive
        max: Maximium number, inclusive
        fragment: Add "?fragment=true" to return the fact as a sentence fragment that can be easily included as part of a larger sentence. This means that the first word is lowercase and ending punctuation is omitted. For trivia and math, a noun phrase is returned that can be used in a sentence like “We now have more users than [fact as fragment]!”.
        json: Specify "true" to return result as JSON instead of plaintext.
        
    """
    url = f"https://numbersapi.p.rapidapi.com/random/{type}"
    querystring = {}
    if min:
        querystring['min'] = min
    if max:
        querystring['max'] = max
    if fragment:
        querystring['fragment'] = fragment
    if json:
        querystring['json'] = json
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "numbersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_trivia_fact(number: str, fragment: str='True', notfound: str='floor', json: str='True', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a trivia fact about a number"
    number: The integer of interest
        fragment: Add "?fragment=true" to return the fact as a sentence fragment that can be easily included as part of a larger sentence. This means that the first word is lowercase and ending punctuation is omitted. For trivia and math, a noun phrase is returned that can be used in a sentence like “We now have more users than [fact as fragment]!”.
        notfound: Specifies what to return if the number is not found. Value can be "default" (to return a canned message), "floor" (to round down to the largest number that does have an associated fact, and return that fact), or "ceil" (which is like floor but rounds up to the smallest number that has an associated fact).
        json: Specify "true" to return result as JSON instead of plaintext.
        
    """
    url = f"https://numbersapi.p.rapidapi.com/{number}/trivia"
    querystring = {}
    if fragment:
        querystring['fragment'] = fragment
    if notfound:
        querystring['notfound'] = notfound
    if json:
        querystring['json'] = json
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "numbersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_year_fact(year: str, fragment: str='True', json: str='True', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a fact about a year"
    year: The year of interest
        fragment: Add "?fragment=true" to return the fact as a sentence fragment that can be easily included as part of a larger sentence. This means that the first word is lowercase and ending punctuation is omitted. For trivia and math, a noun phrase is returned that can be used in a sentence like “We now have more users than [fact as fragment]!”.
        json: Specify "true" to return result as JSON instead of plaintext.
        
    """
    url = f"https://numbersapi.p.rapidapi.com/{year}/year"
    querystring = {}
    if fragment:
        querystring['fragment'] = fragment
    if json:
        querystring['json'] = json
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "numbersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

