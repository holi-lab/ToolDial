import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def zip_4_demographic_append(d_zip4: str, d_zip: str, output: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Zip+4 Demographic Append"
    d_zip4: A 4 digit zip+4 in the USA.
        d_zip: A 5 digit zip code in the USA.
        output: Output
        
    """
    url = f"https://ideas2it-lymbix.p.rapidapi.com/qdf.php"
    querystring = {'d_zip4': d_zip4, 'd_zip': d_zip, }
    if output:
        querystring['output'] = output
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-lymbix.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def zip_demographic_append(d_zip: str, output: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Zip Demographic Append"
    d_zip: Zip Code
        output: Output
        
    """
    url = f"https://ideas2it-lymbix.p.rapidapi.com/qdf.php"
    querystring = {'d_zip': d_zip, }
    if output:
        querystring['output'] = output
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-lymbix.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def demographic(d_city: str, d_state: str, d_zip: str, d_first: str, d_last: str, d_fulladdr: str, d_long: str='-122.084', output: str='json', d_email: str='john.doe@domain.com', d_phone: str='9190014620', d_lat: str='37.386', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Demographic"
    d_city: A city in the USA.
        d_state: A city in the USA
        d_zip: A 5 digit zip code in the USA.
        d_first: First Name
        d_last: Last Name
        d_fulladdr: Entire house number + street + suite
        d_long: Longitude of the location
        output: Output format
        d_email: Valid email address
        d_phone: Phone Number
        d_lat: Latitude of the location
        
    """
    url = f"https://ideas2it-lymbix.p.rapidapi.com/qdf.php"
    querystring = {'d_city': d_city, 'd_state': d_state, 'd_zip': d_zip, 'd_first': d_first, 'd_last': d_last, 'd_fulladdr': d_fulladdr, }
    if d_long:
        querystring['d_long'] = d_long
    if output:
        querystring['output'] = output
    if d_email:
        querystring['d_email'] = d_email
    if d_phone:
        querystring['d_phone'] = d_phone
    if d_lat:
        querystring['d_lat'] = d_lat
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-lymbix.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def email_append(d_email: str, d_first: str, d_last: str, d_fulladdr: str, d_zip: str, d_state: str, d_city: str, output: str='json', d_phone: str='9190014620', d_lat: str='37.386', d_long: str='-122.084', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Email Append"
    d_email: Email Address
        d_first: First Name
        d_last: Last Name
        d_fulladdr: Entire house number + street + suite
        d_zip: A 5 digit zip code in the USA.
        d_state: State Name
        d_city: A city in the USA
        output: JSON/XML
        d_phone: Phone Number
        d_lat: Latitude of the location
        d_long: Longitude of the location
        
    """
    url = f"https://ideas2it-lymbix.p.rapidapi.com/qdf.php"
    querystring = {'d_email': d_email, 'd_first': d_first, 'd_last': d_last, 'd_fulladdr': d_fulladdr, 'd_zip': d_zip, 'd_state': d_state, 'd_city': d_city, }
    if output:
        querystring['output'] = output
    if d_phone:
        querystring['d_phone'] = d_phone
    if d_lat:
        querystring['d_lat'] = d_lat
    if d_long:
        querystring['d_long'] = d_long
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-lymbix.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def phone_append(d_city: str, d_long: str, d_state: str, d_zip: str, output: str, d_phone: str, d_first: str, d_last: str, d_fulladdr: str, d_lat: str='37.386', d_email: str='john.doe@domain.com', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Phone Append"
    d_city: A city in the USA.
        d_long: Longitude of the location
        d_state: A city in the USA
        d_zip: A 5 digit zip code in the USA.
        output: Output
        d_phone: Phone Number
        d_first: First Name
        d_last: Last Name
        d_fulladdr: Entire house number + street + suite
        d_lat: Latitude of the location
        d_email: Valid Email Address
        
    """
    url = f"https://ideas2it-lymbix.p.rapidapi.com/qdf.php"
    querystring = {'d_city': d_city, 'd_long': d_long, 'd_state': d_state, 'd_zip': d_zip, 'output': output, 'd_phone': d_phone, 'd_first': d_first, 'd_last': d_last, 'd_fulladdr': d_fulladdr, }
    if d_lat:
        querystring['d_lat'] = d_lat
    if d_email:
        querystring['d_email'] = d_email
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ideas2it-lymbix.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

