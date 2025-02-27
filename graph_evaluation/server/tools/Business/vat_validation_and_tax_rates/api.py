import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def vat_validation_api_endpoint(vat_number: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Validate Endpoint takes a VAT number in the request, checks if that number is valid, and if it is valid, returns addition details about the company associated with that VAT number."
    vat_number: The VAT number to validate.
        
    """
    url = f"https://vat-validation-and-tax-rates.p.rapidapi.com/v1/validate"
    querystring = {'vat_number': vat_number, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vat-validation-and-tax-rates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def vat_rates_categories_api_endpoint(country_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The /categories/ endpoint makes it easy to get the latest VAT rates, including the reduced rates for certain categories, for a specific country."
    country_code: The two letter ISO 3166-1 alpha-2 code of the country in which the transaction takes place.
        
    """
    url = f"https://vat-validation-and-tax-rates.p.rapidapi.com/v1/categories"
    querystring = {'country_code': country_code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vat-validation-and-tax-rates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def vat_price_calculation_api_endpoint(amount: str, country_code: str, vat_category: str='books', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Calculate Endpoint makes it easy to calculate a VAT compliant price given a country and price, as well as such optional values as the type of goods."
    amount: The amount that you would like to get the VAT amount for or from.
        country_code: The two letter [ISO 3166-1 alpha-2](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2) code of the country in which the transaction takes place.
        vat_category: Some countries give reduced VAT rates for certain categories of goods. If you pass the VAT category in the request, the API will check if there is a reduced VAT rate for that country and category. For example, below is a request to get the VAT for 500 EUR worth of books in Germany
        
    """
    url = f"https://vat-validation-and-tax-rates.p.rapidapi.com/v1/calculate"
    querystring = {'amount': amount, 'country_code': country_code, }
    if vat_category:
        querystring['vat_category'] = vat_category
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vat-validation-and-tax-rates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

