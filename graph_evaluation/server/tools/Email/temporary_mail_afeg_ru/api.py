import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def create(email: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "### Create a mailbox
		The ***"email"*** parameter is what is before ***"@"***  Important: Username length cannot be less than 5 and greator than 15 !!!
		The  ***"domain"***  parameter is what is after the ***"@"*** Required parameter. One from the list obtained by the  ***"Get domain list"*** request.
		Параметр ***"email"***  -  это то, что стоит перед   ***"@"***   Обязательно: количество символов от 5 до 15 !!!
		Параметр ***"domain"***  -  это то, что стоит после  ***"@"***   Один из списка, полученного в запросе ***" Get domain list"***."
    
    """
    url = f"https://temporary-mail-afeg-ru.p.rapidapi.com/api/email/{email}/request"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "temporary-mail-afeg-ru.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_domain_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "### Get domain list"
    
    """
    url = f"https://temporary-mail-afeg-ru.p.rapidapi.com/api/domains/request"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "temporary-mail-afeg-ru.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_mail(email: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "### Get mail content
		***"email"*** -  required parameter. Full name of the mailbox, including the domain (example: email@afeg.ru). 
		
		***"email"*** - обязательный параметр. Полное имя почтового ящика, включая домен (пример: email@afeg.ru)."
    
    """
    url = f"https://temporary-mail-afeg-ru.p.rapidapi.com/api/messages/{email}/request"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "temporary-mail-afeg-ru.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

