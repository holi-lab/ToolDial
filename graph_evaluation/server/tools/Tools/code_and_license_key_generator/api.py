import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def splice(code_string: str, step_count: int, delimiter_string: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Summary**
		Used to interpolate a delimiter pattern into a string at a specified regular interval.
		
		**Parameters**
		- code_string: the string of text you wish to interpolate a delimiter into
		- delimiter_string: the delimiter itself
		- step_count: where in the code_string you want to insert the delimiter. It gets inserted ever nth step
		
		**Example**
		- code_string: fioewiof349f83j4fj34jf93498
		- delimiter:  ==
		- step_count: 4
		- result: fioe==wiof==349f==83j4==fj34==jf93==498"
    
    """
    url = f"https://code-and-license-key-generator.p.rapidapi.com/v1/splice/{code_string}/{delimiter_string}/{step_count}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "code-and-license-key-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_codes(type: str, length: int, step: int, quantity: int, delimiter: int=2, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Summary**
		Bulk generate quantities of generic codes in different formats. Cryptographically selected random numbers included in the strings created.
		
		**Parameters**
		- type: what types of character should be included in each code string. Valid values: ( alphanumeric,   lower_alphanumeric,  all_alphanumeric, alpha,  upper_alpha,  lower_alpha,  all_alpha,  numeric)
		- quantity: the quantity of codes to generate
		- length: the number of characters to include in each code string
		- step: where to insert the delimiter (delimiters are inserted every nth step)
		- delimiter (optional): default is a dash '-'"
    
    """
    url = f"https://code-and-license-key-generator.p.rapidapi.com/v1/codes/{type}/{quantity}/{length}/{step}/{delimiter}"
    querystring = {}
    if delimiter:
        querystring['delimiter'] = delimiter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "code-and-license-key-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def uuids(quantity: int, uuid_version: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Summary**
		Bulk generates universally [unique identifier (UUID) codes](https://www.uuidtools.com/what-is-uuid).  This endpoint can create type 1 (MAC address + timestamp-based) and type 4  (random-based) UUIDs.
		
		**Parameters**
		- uuid_version: The version of UUID that you wish to acquire codes for
		- quantity: the number of codes you wish to generate"
    
    """
    url = f"https://code-and-license-key-generator.p.rapidapi.com/v1/uuid/{uuid_version}/{quantity}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "code-and-license-key-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def patterns(quantity: int, pattern_string: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Summary**
		Bulk generates codes according to the pattern string specified.
		
		**Parameters**
		- quantity: the number of codes you wish to 
		- pattern_string: a string of numbers used to identify what  type of character should be used next in the string. Numbers are mapped as follows:
		  1. upper alphanumeric
		  2. lower alphanumeric
		  3. all alphanumeric (integer or alphabet character either upper or lower or integer)
		  4. upper alpha (uppercase alphabet)
		  5. lower alpha (lowercase alphabet)
		  6. all alpha (either upper or lower case alphabet)
		  7. integer
		  8. special characters
		
		**Example**
		123877 would be interpreted as:
		-  'upper alpha' (1)
		-  'lower alpha' (2)
		-  'all alphanumeric' (3)
		- 'special' (8)
		- 'integer' (7)
		- 'integer' (7)
		Which  could potentially generate a code like this: Ab3_89"
    
    """
    url = f"https://code-and-license-key-generator.p.rapidapi.com/v1/pattern/{quantity}/{pattern_string}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "code-and-license-key-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

