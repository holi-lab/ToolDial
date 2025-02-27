import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def sortz_data_sorting(sort: str, list: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using the **/sortz** endpoint will help when the data needs sorting.  After you've cleaned your data it may be time to sort it alphabetically or in reverse.
		
		Currently available scrubbing options are:  alphabetize, alphabetizereverse
		
		This is a small set of options, but can be built out upon request.  Object and Array sorting will be added as demand requires."
    
    """
    url = f"https://zzzapio-utilities.p.rapidapi.com/scrubz/{sort}"
    querystring = {'list': list, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zzzapio-utilities.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def scrubz_data_validating_formatting(list: str, text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using the **/scrubz** endpoint will help when the data needs validated (or scrubbed) and or when it is not formatted correctly.  Scrubbing is crucial to database integrity as well as UX design. Data that comes in poorly formatted, too large, too small or altogether wrong can plague or even break front-end and back-end logic.
		
		Currently available scrubbing options are:  required, nospace, number, alpha, alphanumeric, alphanumspace, alphaspace,datetime, email, maxlen:n, minlen:n, setlen:n, islte:n, isgte:n, uppercase, lowercase, encrypt, decrypt, encrypthash, hash, base64on, base64off, number, trim
		
		*scrub:n, where n is a number, is the format for defining the length to be tested.  For example,  using **maxlen:12**  requires the data to have a maximum length of 12 characters. 
		
		Use as many scrubbing options as you need to clean and validate dirty data."
    
    """
    url = f"https://zzzapio-utilities.p.rapidapi.com/scrubz"
    querystring = {'list': list, 'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zzzapio-utilities.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def formatz_data_formatting(format: str, text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using the **/formatz** endpoint will help when the data is not formatted correctly. This is also where data can pass through encrypt/decrypt for secure data.
		
		Currently available formatting options are:  uppercase, lowercase, encrypt, decrypt, encrypthash, hash, base64on, base64off, number, trim"
    
    """
    url = f"https://zzzapio-utilities.p.rapidapi.com/formatz/{format}"
    querystring = {'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zzzapio-utilities.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

