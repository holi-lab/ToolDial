import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def extract(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint extracts complete names from unstructured text very fast. By using a combination of 1.712.459 first names and 5.238.897 last names we can identify and extract complete names with an extremely high precision and accuracy. The name extraction endpoint is very useful for software developers and publishers. Extract names from unstructured text with an extremely high precision and accuracy. Use our API and Web App to extract names from unstructured text or documents."
    
    """
    url = f"https://name-parser1.p.rapidapi.com/extract/"
    querystring = {'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "name-parser1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def parse(name: str, validate: bool=None, ip: str='66.249.64.85', country_code: str='DE', email: str='johnsmith@hotmail.com', gender_type: str='gender', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint parses a complete name or email address and returns the first name, middle names and last name. Additionally it also returns the nickname, salutation, title, email provider and the most likely nationality of the given name. The name parsing endpoint is very useful for online businesses and services. It shortens forms, validates names and enriches existing customer database with their gender, salutation and nationality. Use our API and Web App to parse names into useful components."
    name: The complete name that you want to parse. Depending on your subscription you can query multiple names in a single request by making an array out of it.
Example: name[]=John Doe&name[]=Sarah Jones
        ip: If you don't know the country code you can always provide an IP address. We convert the IP address to a country_code internally. If the name parameter contains multiple names then the ip should also be an array.
        country_code: If you know the country code associated with the name then you can provide it. By providing the country code the accuracy of the gender will be higher. The country code should be 2 characters according to the ISO 3166-2 country code specification. If the name parameter contains multiple names then the country_code should also be an array.
        gender_type: Different type of kind can be used as a value for gender_formatted. By default the gender field returns \\\"male\\\" or \\\"female\\\". You can change the formatting from sex (default) to gender, marital, birth and slang.
        
    """
    url = f"https://name-parser1.p.rapidapi.com/parse/"
    querystring = {'name': name, }
    if validate:
        querystring['validate'] = validate
    if ip:
        querystring['ip'] = ip
    if country_code:
        querystring['country_code'] = country_code
    if email:
        querystring['email'] = email
    if gender_type:
        querystring['gender_type'] = gender_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "name-parser1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generate(country_code: str='DE', gender: str=None, ip: str='66.249.64.85', results: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint generates names by combining a random first name and a random last name for any given country code. Additionally the endpoint also generates a fictional email address and strong password making it a great solution to create development databases. The name generator endpoint is very useful for software developers. Instead of developing software with real data its more secure to use fictional accounts instead. Use our API and Web App to generate fictional accounts including email address, strong password, gender and country details."
    country_code: Use the country_code parameter if you want to generate random names for a specific country. The country code should be 2 characters according to the ISO 3166-2 country code specification.
        gender: Different type of kind can be used as a value for gender_formatted. By default the gender field returns \\\"male\\\" or \\\"female\\\". You can change the formatting from sex (default) to gender, marital, birth and slang.
        ip: If you don't know the country code you can always provide an IP address. We convert the IP address to a country_code internally. If the name parameter contains multiple names then the ip should also be an array.
        results: The number of generated names you want to receive in the response object. For example: 3 will give you 3 different generated name objects. The maximum number of queries depends on you subscription.
        
    """
    url = f"https://name-parser1.p.rapidapi.com/generate/"
    querystring = {}
    if country_code:
        querystring['country_code'] = country_code
    if gender:
        querystring['gender'] = gender
    if ip:
        querystring['ip'] = ip
    if results:
        querystring['results'] = results
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "name-parser1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

