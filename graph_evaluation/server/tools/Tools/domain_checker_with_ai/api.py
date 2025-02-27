import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def domain_name_suggester(words: str, prefixes: str='ultra', tlds: str='com,net,org', suffixes: str='ify,x', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "AI powered endpoint to easily generate new and unique domain name suggestions by sending a GET request to the specified endpoint with base words and other optional parameters. It uses a combination of user input and predefined AI trained models to generate the suggestions, making it a powerful tool for finding the perfect domain name.
		**Parameters and its description**
		
		1. words: base words (like shop, market etc.)
		2. tlds: TLDs (like com,net,in etc.)
		3. prefixes: prefixes which can be used in the domain name (like ultra, simply etc.)
		4. suffixes: suffixes which can be used in the domain name (like ify, x etc.)
		
		**Note**
		
		- Except `words` parameter, all other parameters are optional. 
		- All parameter supports comma-separated multi values like `car,bike,plane` etc.
		
		**Constrained by a strategic blueprint**
		
		1. For Ultra: Each parameter is limited to **no more than five comma-separated inputs**. Any additional values will be disregarded and not used in processing.
		2. For Mega: Each parameter is limited to **no more than ten comma-separated inputs**. Any additional values will be disregarded and not used in processing.
		
		Your input parameters will allow our AI model to generate results that are tailored to your specific needs."
    
    """
    url = f"https://domain-checker-with-ai.p.rapidapi.com/domain/suggest"
    querystring = {'words': words, }
    if prefixes:
        querystring['prefixes'] = prefixes
    if tlds:
        querystring['tlds'] = tlds
    if suffixes:
        querystring['suffixes'] = suffixes
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "domain-checker-with-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dns_details(domain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To use this endpoint, you need to send a GET request to the endpoint with the `domain` field in the query parameter. The value of this field should be the domain name for which you want to retrieve DNS information.
		The endpoint will return a JSON response with the complete DNS information for the domain as a dictionary. If the DNS information could not be retrieved, the endpoint will return a `400 Bad Request` error with an `'error'` field containing an error message."
    
    """
    url = f"https://domain-checker-with-ai.p.rapidapi.com/domain/dns"
    querystring = {'domain': domain, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "domain-checker-with-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def domain_whois(domain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To use this endpoint, you need to send a GET request to the endpoint with the `domain` field in the query parameter. The value of this field should be the domain name for which you want to retrieve WHOIS information.
		The endpoint will return a JSON response with the WHOIS information for the domain as a dictionary. If the WHOIS information could not be retrieved, the endpoint will return a `400 Bad Request` error with an `'error'` field containing an error message."
    
    """
    url = f"https://domain-checker-with-ai.p.rapidapi.com/domain/whois"
    querystring = {'domain': domain, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "domain-checker-with-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def domain_validation(domain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to validate a domain name by checking if it is registered and has an IP address associated with it.
		To use this endpoint, you need to send a GET request to the endpoint with the `domain` field in the query parameter. The value of this field should be the domain name you want to validate.
		The endpoint will return a JSON response with a `status` field set to `'valid'` if the domain is valid, and an `'ip-address'` field containing the IP address of the domain. If the domain is not valid, the endpoint will return a `400 Bad Request` error with an `'error'` field set to `'invalid domain'`."
    
    """
    url = f"https://domain-checker-with-ai.p.rapidapi.com/domain/validate"
    querystring = {'domain': domain, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "domain-checker-with-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

