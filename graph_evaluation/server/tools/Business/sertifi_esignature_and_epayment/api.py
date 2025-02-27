import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_folderid_participants(folderid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    folderid: The id of the folder to retrieve data for.
        
    """
    url = f"https://sertifi-esignature-and-epayment.p.rapidapi.com/v1/{folderid}/participants"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sertifi-esignature-and-epayment.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getfolder(folderid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    folderid: The id of the folder to retrieve data for.
        
    """
    url = f"https://sertifi-esignature-and-epayment.p.rapidapi.com/v1/folders/{folderid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sertifi-esignature-and-epayment.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_folders_folderid_summary_documentid(folderid: str, documentid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    folderid: The id of the folder to retrieve data for.
        documentid: The id of the document to retrieve data for.
        
    """
    url = f"https://sertifi-esignature-and-epayment.p.rapidapi.com/v1/folders/{folderid}/summary/{documentid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sertifi-esignature-and-epayment.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_users_email(email: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    email: Email of the user
        
    """
    url = f"https://sertifi-esignature-and-epayment.p.rapidapi.com/v1/users/{email}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sertifi-esignature-and-epayment.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_folderid_payments(folderid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    folderid: The id of the folder to retrieve data for.
        
    """
    url = f"https://sertifi-esignature-and-epayment.p.rapidapi.com/v1/{folderid}/payments"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sertifi-esignature-and-epayment.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_folderid_invites(folderid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    folderid: The id of the folder to send invites for.
        
    """
    url = f"https://sertifi-esignature-and-epayment.p.rapidapi.com/v1/{folderid}/invites"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sertifi-esignature-and-epayment.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_folderid_payments_paymentid(paymentid: str, folderid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    paymentid: The id of the payment to retrieve data for.
        folderid: The id of the folder to retrieve data for.
        
    """
    url = f"https://sertifi-esignature-and-epayment.p.rapidapi.com/v1/{folderid}/payments/{paymentid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sertifi-esignature-and-epayment.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_folders_folderid_content(folderid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    folderid: The folder id of the folder to download
        
    """
    url = f"https://sertifi-esignature-and-epayment.p.rapidapi.com/v1/folders/{folderid}/content"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sertifi-esignature-and-epayment.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

