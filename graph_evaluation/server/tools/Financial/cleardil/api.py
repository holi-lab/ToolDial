import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def fetchindividualcustomer(authorization: str, customer_a_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://cleardil1.p.rapidapi.com/customers/{customer_a_id}"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cleardil1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetchallindividualcustomers(authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://cleardil1.p.rapidapi.com/customers"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cleardil1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def viewoverallscreeningresult(authorization: str, content_type: str, customer_a_id: str, screening_a_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://cleardil1.p.rapidapi.com/customers/{customer_a_id}/screenings/{screening_a_id}"
    querystring = {'Authorization': authorization, 'Content-Type': content_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cleardil1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetchanote(content_type: str, authorization: str, customer_a_id: str, note_a_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://cleardil1.p.rapidapi.com/customers/{customer_a_id}/notes/{note_a_id}"
    querystring = {'Content-Type': content_type, 'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cleardil1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def screeningstats(authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://cleardil1.p.rapidapi.com/stats/screenings"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cleardil1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def viewallscreeningsresult(authorization: str, content_type: str, customer_a_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://cleardil1.p.rapidapi.com/customers/{customer_a_id}/screenings"
    querystring = {'Authorization': authorization, 'Content-Type': content_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cleardil1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def customerstats(authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://cleardil1.p.rapidapi.com/stats/customers"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cleardil1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetchallnotes(authorization: str, content_type: str, customer_a_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://cleardil1.p.rapidapi.com/customers/{customer_a_id}/notes"
    querystring = {'Authorization': authorization, 'Content-Type': content_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cleardil1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def searchallyourconfirmedmatches(content_type: str, authorization: str, validation_result: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://cleardil1.p.rapidapi.com/search/matches"
    querystring = {'Content-Type': content_type, 'Authorization': authorization, 'validation_result': validation_result, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cleardil1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetchdriving_licensedocument(authorization: str, customer_a_id: str, document_a_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://cleardil1.p.rapidapi.com/customers/{customer_a_id}/documents/{document_a_id}"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cleardil1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetchriskprofile(content_type: str, authorization: str, customer_a_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Create individual customer"
    
    """
    url = f"https://cleardil1.p.rapidapi.com/customers/{customer_a_id}/risk_profile"
    querystring = {'Content-Type': content_type, 'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cleardil1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def downloadtheattacheddriving_licensefromthedocument(authorization: str, customer_a_id: str, document_a_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://cleardil1.p.rapidapi.com/customers/{customer_a_id}/documents/{document_a_id}/download"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cleardil1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetchoneassociationbelongingtoamatch(content_type: str, authorization: str, asso_id: str, match_id: str, customer_a_id: str, screening_a_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://cleardil1.p.rapidapi.com/customers/{customer_a_id}/screenings/{screening_a_id}/matches/{match_id}/associations/{asso_id}"
    querystring = {'Content-Type': content_type, 'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cleardil1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetchallassociationsbelongingtoamatch(authorization: str, content_type: str, customer_a_id: str, screening_a_id: str, match_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://cleardil1.p.rapidapi.com/customers/{customer_a_id}/screenings/{screening_a_id}/matches/{match_id}/associations"
    querystring = {'Authorization': authorization, 'Content-Type': content_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cleardil1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def searchscreeningsdone(authorization: str, content_type: str, status: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://cleardil1.p.rapidapi.com/search/screenings"
    querystring = {'Authorization': authorization, 'Content-Type': content_type, 'status': status, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cleardil1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def searchverificationsdone(authorization: str, content_type: str, status: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://cleardil1.p.rapidapi.com/search/verifications"
    querystring = {'Authorization': authorization, 'Content-Type': content_type, 'status': status, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cleardil1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetchaverification(authorization: str, content_type: str, verification_a_id: str, customer_a_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://cleardil1.p.rapidapi.com/customers/{customer_a_id}/verifications/{verification_a_id}"
    querystring = {'Authorization': authorization, 'Content-Type': content_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cleardil1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetchamatch(authorization: str, content_type: str, customer_a_id: str, match_1_id: str, screening_a_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://cleardil1.p.rapidapi.com/customers/{customer_a_id}/screenings/{screening_a_id}/matches/{match_1_id}"
    querystring = {'Authorization': authorization, 'Content-Type': content_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cleardil1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetchallmatches(content_type: str, authorization: str, screening_a_id: str, customer_a_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://cleardil1.p.rapidapi.com/customers/{customer_a_id}/screenings/{screening_a_id}/matches"
    querystring = {'Content-Type': content_type, 'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cleardil1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetchallverifications(content_type: str, authorization: str, customer_a_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://cleardil1.p.rapidapi.com/customers/{customer_a_id}/verifications"
    querystring = {'Content-Type': content_type, 'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cleardil1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

