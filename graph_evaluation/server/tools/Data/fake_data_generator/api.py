import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def random_address(size: str='5', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With this enpoints you can manage the generation of fake addresses and retrieve them by id."
    
    """
    url = f"https://fake-data-generator1.p.rapidapi.com/address/random_address"
    querystring = {}
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fake-data-generator1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_custom(layout: str, size: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint generate a list of fake  data ustomizing the layout setting the number and type.
		
		You can define fake data layout with de 'layout' param.
		
		The structure of this param:
		
		```
		{'fields': 	{
						'date': {'type': 'DATE_AND_HOUR'},
						'name': {'type': 'NAME_FULL_NAME'},
						'purchase': {
										'type': 'COMPOUND', 
										'fields': {
													'company': {'type': 'COMPANY_NAME'},
													'amount': {'type': 'CURRENCY_PRICE'}
												   }
									}
					}
		}
		```
		
		With 'type': 'COMPOUND', you can add nodes, tha max depth is 3.
		
		In regard to the diferents type of data, you can use this types:
		
		```
		'COUNTRY'
		'COUNTRY_CODE'
		'CITY'
		'STREET_ADDRESS'
		'BUILDING_NUMBER'
		'POSTAL_CODE'
		'COMPANY_NAME'
		'COMPANY_SUFFIX'
		'COMPANY_BUSINESS'
		'COMPANY_PHRASE'
		'CREDIT_CARD_EXPIRE'
		'CREDIT_CARD_NUMBER'
		'CREDIT_CARD_PROVIDER'
		'CREDIT_CARD_SECURITY_CODE'
		'CURRENCY_PRICE'
		'CURRENCY_CODE_NAME'
		'CURRENCY_CRYPTO_CODE_NAME'
		'NAME_PREFIX_NAME'
		'NAME_FIRST_NAME_MALE'
		'NAME_FIRST_NAME_FEMALE'
		'NAME_LASTNAME'
		'DATE'
		'DATE_AND_HOUR'
		'DATE_CENTURY'
		'DATE_THIS_DECADE'
		'DATE_THIS_MONTH'
		'INT'
		'PARAGRAPH'
		'PARAGRAPH_LOREMIPSUM'
		'SENTENCE'
		'SENTENCE_LOREMIPSUM'
		'BBAN'
		'IBAN'
		'SWIFT'
		'SWIFT11'
		'TEXT'
		'TEXT_LOREMIPSUM'
		'URL'
		'WORD'
		```"
    
    """
    url = f"https://fake-data-generator1.p.rapidapi.com/custom/random_custom"
    querystring = {'layout': layout, }
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fake-data-generator1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_custom_by_id(is_id: int, layout: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve  fake  data banks by id customizing the layout setting the number and type.
		
		You can define fake data layout with de 'layout' param.
		
		The structure of this param:
		
		```
		{'fields': 	{
						'date': {'type': 'DATE_AND_HOUR'},
						'name': {'type': 'NAME_FULL_NAME'},
						'purchase': {
										'type': 'COMPOUND', 
										'fields': {
													'company': {'type': 'COMPANY_NAME'},
													'amount': {'type': 'CURRENCY_PRICE'}
												   }
									}
					}
		}
		```
		
		With 'type': 'COMPOUND', you can add nodes, tha max depth is 3.
		
		In regard to the diferents type of data, you can use this types:
		
		```
		'COUNTRY'
		'COUNTRY_CODE'
		'CITY'
		'STREET_ADDRESS'
		'BUILDING_NUMBER'
		'POSTAL_CODE'
		'COMPANY_NAME'
		'COMPANY_SUFFIX'
		'COMPANY_BUSINESS'
		'COMPANY_PHRASE'
		'CREDIT_CARD_EXPIRE'
		'CREDIT_CARD_NUMBER'
		'CREDIT_CARD_PROVIDER'
		'CREDIT_CARD_SECURITY_CODE'
		'CURRENCY_PRICE'
		'CURRENCY_CODE_NAME'
		'CURRENCY_CRYPTO_CODE_NAME'
		'NAME_PREFIX_NAME'
		'NAME_FIRST_NAME_MALE'
		'NAME_FIRST_NAME_FEMALE'
		'NAME_LASTNAME'
		'DATE'
		'DATE_AND_HOUR'
		'DATE_CENTURY'
		'DATE_THIS_DECADE'
		'DATE_THIS_MONTH'
		'INT'
		'PARAGRAPH'
		'PARAGRAPH_LOREMIPSUM'
		'SENTENCE'
		'SENTENCE_LOREMIPSUM'
		'BBAN'
		'IBAN'
		'SWIFT'
		'SWIFT11'
		'TEXT'
		'TEXT_LOREMIPSUM'
		'URL'
		'WORD'
		```"
    
    """
    url = f"https://fake-data-generator1.p.rapidapi.com/custom/random_custom_by_id"
    querystring = {'id': is_id, 'layout': layout, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fake-data-generator1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_bank(size: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint generate a list of fake banks"
    
    """
    url = f"https://fake-data-generator1.p.rapidapi.com/bank/random_bank"
    querystring = {}
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fake-data-generator1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_company(size: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint generate a list of fake companies"
    
    """
    url = f"https://fake-data-generator1.p.rapidapi.com/company/random_company"
    querystring = {}
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fake-data-generator1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_company_by_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve fake companies by id."
    
    """
    url = f"https://fake-data-generator1.p.rapidapi.com/company/random_company_by_id"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fake-data-generator1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_credit_card(size: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint generate a list of fake credit cards"
    
    """
    url = f"https://fake-data-generator1.p.rapidapi.com/credit_card/random_credit_card"
    querystring = {}
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fake-data-generator1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_crypto(size: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint generate a list of fake crypto assets"
    
    """
    url = f"https://fake-data-generator1.p.rapidapi.com/crypto/random_crypto"
    querystring = {}
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fake-data-generator1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_crypto_by_id(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve fake crypto assets by id."
    
    """
    url = f"https://fake-data-generator1.p.rapidapi.com/crypto/random_crypto_by_id"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fake-data-generator1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_post(size: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint generate a list of fake posts"
    
    """
    url = f"https://fake-data-generator1.p.rapidapi.com/post/random_post"
    querystring = {}
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fake-data-generator1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_address_by_id(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a fake address by id"
    
    """
    url = f"https://fake-data-generator1.p.rapidapi.com/address/random_address_by_id"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fake-data-generator1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_profiles(size: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint generate a list of fake user profiles"
    
    """
    url = f"https://fake-data-generator1.p.rapidapi.com/user/random_profiles"
    querystring = {}
    if size:
        querystring['size'] = size
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fake-data-generator1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_profile_by_id(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve fake profiles by id."
    
    """
    url = f"https://fake-data-generator1.p.rapidapi.com/user/random_profile_by_id"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fake-data-generator1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_bank_by_id(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve fake banks by id."
    
    """
    url = f"https://fake-data-generator1.p.rapidapi.com/bank/random_bank_by_id"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fake-data-generator1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_credit_card_by_id(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve fake credit cards by id."
    
    """
    url = f"https://fake-data-generator1.p.rapidapi.com/credit_card/random_credit_card_by_id"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fake-data-generator1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_post_by_id(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve fake post by id."
    
    """
    url = f"https://fake-data-generator1.p.rapidapi.com/post/random_post_by_id"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fake-data-generator1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

