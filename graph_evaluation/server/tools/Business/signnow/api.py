import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def embedded_signing_get_role_ids(document_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns details of specific document
		
		The Document object contains:
		
		- Metadata: file name, size, extension, ID;
		- Fields, elements (texts, checks and signatures),
		- Invites, status of the invites,
		- Roles,
		- Timestamps (date created, date updated)
		
		Document is a fundamental object of every e-signature operation. It’s impossible to send an invite without selecting or [uploading a document](https://docs.signnow.com/docs/signnow/reference/operations/create-a-document) first. In signNow you can add other objects to the document called [fillable fields](https://docs.signnow.com/docs/signnow/branches/v1.2/reference/operations/update-a-document-document-id-adds-fields-to-a-document). When the document is signed, it can be [downloaded in PDF file format](https://docs.signnow.com/docs/signnow/reference/operations/get-a-document-download). Documents in signNow can be [turned into a template](https://docs.signnow.com/docs/signnow/reference/operations/create-a-template). API also allows you to create [event subscriptions](https://docs.signnow.com/docs/signnow/reference/operations/create-a-api-v-2-event) triggered by a specific document."
    
    """
    url = f"https://signnowapi-signnow.p.rapidapi.com/document/{document_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signnowapi-signnow.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_field_and_field_invite_id(document_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint returns field and field invite ids."
    document_id: id of the document
        
    """
    url = f"https://signnowapi-signnow.p.rapidapi.com/document/{document_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signnowapi-signnow.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_role_ids(document_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns details of specific document
		
		The Document object contains:
		
		- Metadata: file name, size, extension, ID;
		- Fields, elements (texts, checks and signatures),
		- Invites, status of the invites,
		- Roles,
		- Timestamps (date created, date updated)
		
		Document is a fundamental object of every e-signature operation. It’s impossible to send an invite without selecting or [uploading a document](https://docs.signnow.com/docs/signnow/reference/operations/create-a-document) first. In signNow you can add other objects to the document called [fillable fields](https://docs.signnow.com/docs/signnow/branches/v1.2/reference/operations/update-a-document-document-id-adds-fields-to-a-document). When the document is signed, it can be [downloaded in PDF file format](https://docs.signnow.com/docs/signnow/reference/operations/get-a-document-download). Documents in signNow can be [turned into a template](https://docs.signnow.com/docs/signnow/reference/operations/create-a-template). API also allows you to create [event subscriptions](https://docs.signnow.com/docs/signnow/reference/operations/create-a-api-v-2-event) triggered by a specific document."
    
    """
    url = f"https://signnowapi-signnow.p.rapidapi.com/document/{document_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signnowapi-signnow.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_signing_links(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint allows to get all existing signing links and IDs of these signing links."
    
    """
    url = f"https://signnowapi-signnow.p.rapidapi.com/v2/application/signing-links"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signnowapi-signnow.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_role_ids_for_template(template_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint allows to get role ids for a template."
    template_id: ID of the template.
        
    """
    url = f"https://signnowapi-signnow.p.rapidapi.com/document/{template_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signnowapi-signnow.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

