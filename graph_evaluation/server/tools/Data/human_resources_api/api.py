import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_all_employees_payment_cards(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gets all the available records in the bank cards table, up to 900 cards with their information.
		For example:  card number , holder name, and  etc..."
    
    """
    url = f"https://human-resources-api.p.rapidapi.com/api/cards"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "human-resources-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_specific_employee_by_employee_id(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gets one record by a specific employee ID that is unique for each employee in the employees table, up to 900 employees with their information.
		For example:  employee ID , first name, email, and  etc..."
    
    """
    url = f"https://human-resources-api.p.rapidapi.com/api/employees/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "human-resources-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_specific_payment_card_by_card_number(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gets one record by a specific card number that is unique for each card in the bank cards table, up to 900 cards with their information.
		For example:  card number , holder name, and  etc..."
    
    """
    url = f"https://human-resources-api.p.rapidapi.com/api/cards/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "human-resources-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def paginate_through_campany_employees_information(start: int, end: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lets you paginate through all the available records in the employees' table in case you only need a limited number of records for your application, the response includes some metadata information to help you with pagination in the frontend, up to 900 employees with their information.
		For example:  employee ID, first name,  email, and etc..."
    
    """
    url = f"https://human-resources-api.p.rapidapi.com/api/employees/paginate/{start}/{end}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "human-resources-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def paginate_through_employees_payment_cards_infomation(end: int, start: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lets you paginate through all the available records in the bank cards' table in case you only need a limited number of records for your application, the response includes some metadata information to help you with pagination in the frontend, up to 900 bank cards with their information.
		For example:  card number, holder name,  card type, and etc..."
    
    """
    url = f"https://human-resources-api.p.rapidapi.com/api/cards/paginate/{start}/{end}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "human-resources-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_employee_information_associated_with_a_specific_payment_card(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gets employee record associated with a specific card number that is unique for each card, and displays  the employee information in the response.
		For example:  employee ID , first name, and  etc..."
    
    """
    url = f"https://human-resources-api.p.rapidapi.com/api/relation/cards/{is_id}/employee"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "human-resources-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_payment_cards_information_associated_with_a_specific_employee(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gets all bank payment cards records associated with a specific employee ID that is unique for each employee and displays all cards information in the response.
		For example:  card number, expiry date, and etc..."
    
    """
    url = f"https://human-resources-api.p.rapidapi.com/api/relation/employees/{is_id}/cards"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "human-resources-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_campany_employees_information(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gets all the available records in the employees table, up to 900 employees with their information.
		For example:  employee ID, first name,  email, and etc..."
    
    """
    url = f"https://human-resources-api.p.rapidapi.com/api/employees"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "human-resources-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

