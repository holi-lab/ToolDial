import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def fetch_bulk_signing_status(request_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to track the status of the bulk signing requests."
    
    """
    url = f"https://signeasy5.p.rapidapi.com/v3/rs/bulk/sign/{request_id}/track/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signeasy5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_a_signed_envelope_envelope_without_fields(signed_file_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch details of a specific signing request which has been complete by passing the signed_file_id in the path parameter."
    
    """
    url = f"https://signeasy5.p.rapidapi.com/v3/rs/signed/{signed_file_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signeasy5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_all_envelopes_envelope_without_fields(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all the information of the signing requests that were sent out without pre-filling fields."
    
    """
    url = f"https://signeasy5.p.rapidapi.com/v3/rs/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signeasy5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download_a_pending_envelope_envelope_without_fields(pending_file_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Download an incomplete signing request document by passing the pending_file_id in the path parameter."
    
    """
    url = f"https://signeasy5.p.rapidapi.com/v3/rs/{pending_file_id}/download"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signeasy5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download_certificate_of_a_self_signed_document(signed_file_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Download the audit trail document and save it locally at your system."
    
    """
    url = f"https://signeasy5.p.rapidapi.com/v3/me/signed/{signed_file_id}/certificate/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signeasy5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download_a_signed_envelope_with_certificate_embedded_sending(signed_file_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Download a signed envelope along with the audit trail by passing appropriate values to the path parameter
		
		Supported values for **type**: "***split***" and "***merged***"
		
		Supported values for **include_certificate**: (0 and 1) / (***true*** and ***false***) \[Boolean\]"
    
    """
    url = f"https://signeasy5.p.rapidapi.com/v3/embedded/signed/{signed_file_id}/download"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signeasy5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_a_signed_envelope_embedded_sending(signed_file_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch details of a specific signing request which has been complete by passing the signed_file_id in the path parameter."
    
    """
    url = f"https://signeasy5.p.rapidapi.com/v3/rs/embedded/signed/{signed_file_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signeasy5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download_a_pending_envelope(pending_file_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Download the document sent for e-signature using embedded sending by passing the pending_file_id in the path parameter."
    
    """
    url = f"https://signeasy5.p.rapidapi.com/v3/rs/embedded/{pending_file_id}/download/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signeasy5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_a_signed_envelope(signed_file_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch details of a specific signing request which has been complete by passing the signed_file_id in the path parameter."
    
    """
    url = f"https://signeasy5.p.rapidapi.com/v3/rs/envelope/signed/{signed_file_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signeasy5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_signed_id_using_pending_id(pending_file_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch the *signed_file_id* from a *pending_file_id* once a signing request is complete.
		
		The *signed_file_id* is mapped to the ***id*** key in the response.
		
		**NOTE**: This API will return 200 OK only if the signing request is complete/finalised."
    
    """
    url = f"https://signeasy5.p.rapidapi.com/v3/rs/envelope/signed/pending/{pending_file_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signeasy5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download_an_original_from_signed_envelope(signed_file_id: str, original_file_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Download a signed envelope along with the audit trail by passing appropriate values to the path parameter
		
		Supported values for **type**: "***split***" and "***merged***"
		
		Supported values for **include_certificate**: (0 and 1) / (***true*** and ***false***) \[Boolean\]"
    
    """
    url = f"https://signeasy5.p.rapidapi.com/v3/rs/envelope/signed/{signed_file_id}/{original_file_id}/download/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signeasy5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download_pending_envelope_documents_as_merged(pending_file_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Download a particular envelope by passing the pending_file_id in the path parameter."
    
    """
    url = f"https://signeasy5.p.rapidapi.com/v3/rs/envelope/{pending_file_id}/download/merged/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signeasy5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_all_envelopes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch the list of all your envelope data which aren't signed yet."
    
    """
    url = f"https://signeasy5.p.rapidapi.com/v3/rs/envelope/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signeasy5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_all_templates(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetches all templates."
    
    """
    url = f"https://signeasy5.p.rapidapi.com/v3/template/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signeasy5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_a_template(template_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch a specific template."
    
    """
    url = f"https://signeasy5.p.rapidapi.com/v3/template/{template_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signeasy5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_all_originals(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all your uploaded originals in order to retrieve all your document information."
    
    """
    url = f"https://signeasy5.p.rapidapi.com/v3/original/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signeasy5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_an_original(original_file_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch a specific original document details by specifying the original_id in the path parameter."
    
    """
    url = f"https://signeasy5.p.rapidapi.com/v3/original/{original_file_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signeasy5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download_an_original_document(original_file_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Download the original document back to your local machine by specifying the original_id."
    
    """
    url = f"https://signeasy5.p.rapidapi.com/v3/original/{original_file_id}/download"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signeasy5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_user(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve information about the current authorized user. You will receive the user details even if the user has not verified their email address or if the account is not active."
    
    """
    url = f"https://signeasy5.p.rapidapi.com/v3/me/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signeasy5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_an_envelope_envelope_without_fields(pending_file_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List information of a particular signing request without fields by passing the pending_file_id in the path parameter."
    
    """
    url = f"https://signeasy5.p.rapidapi.com/v3/rs/{pending_file_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signeasy5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download_certificate_of_a_signed_envelope(signed_file_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Download a signed envelope along with the audit trail by passing appropriate values to the path parameter
		
		Supported values for **type**: "***split***" and "***merged***"
		
		Supported values for **include_certificate**: (0 and 1) / (***true*** and ***false***) \[Boolean\]"
    
    """
    url = f"https://signeasy5.p.rapidapi.com/v3/rs/envelope/signed/{signed_file_id}/certificate"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signeasy5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_all_signed_envelopes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all the information of your signing requests which have been completed/finalised by all your signers."
    
    """
    url = f"https://signeasy5.p.rapidapi.com/v3/rs/envelope/signed/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signeasy5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download_pending_envelope_documents_as_split(pending_file_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Download pending envelope documents as individual files."
    
    """
    url = f"https://signeasy5.p.rapidapi.com/v3/rs/envelope/{pending_file_id}/download/split"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signeasy5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_a_self_signed_document(signed_file_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch specific details of a self signed document by passing the signed_id in the path."
    
    """
    url = f"https://signeasy5.p.rapidapi.com/v3/me/signed/{signed_file_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signeasy5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_an_envelope(pending_file_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch the list of a particular envelope by passing the pending_file_id in the path parameter."
    
    """
    url = f"https://signeasy5.p.rapidapi.com/v3/rs/envelope/{pending_file_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signeasy5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_all_self_signed_documents(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch all the information from your self signed documents."
    
    """
    url = f"https://signeasy5.p.rapidapi.com/v3/me/signed/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signeasy5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_all_signed_envelope_envelope_without_fields(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all the information of your signing requests which have been completed/finalised by all your signers."
    
    """
    url = f"https://signeasy5.p.rapidapi.com/v3/rs/signed/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signeasy5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download_a_signed_envelope_with_certificate(signed_file_id: str, include_certificate: str, type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Download a signed envelope along with the audit trail by passing appropriate values to the path parameter
		
		Supported values for **type**: "***split***" and "***merged***"
		
		Supported values for **include_certificate**: (0 and 1) / (***true*** and ***false***) \[Boolean\]"
    include_certificate: boolean
        type: 'split' or 'merged'
        
    """
    url = f"https://signeasy5.p.rapidapi.com/v3/signed/{signed_file_id}/download"
    querystring = {'include_certificate': include_certificate, 'type': type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signeasy5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_all_signed_envelopes_embedded_sending(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all the information of your signing requests which have been completed/finalised by all your signers."
    
    """
    url = f"https://signeasy5.p.rapidapi.com/v3/rs/embedded/signed/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signeasy5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download_a_self_signed_document(signed_file_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Download the self signed document and save it locally at your system."
    
    """
    url = f"https://signeasy5.p.rapidapi.com/v3/me/signed/{signed_file_id}/download"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signeasy5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download_certificate_of_a_signed_envelope_embedded_sending(signed_file_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Download a signed envelope along with the audit trail by passing appropriate values to the path parameter
		
		Supported values for **type**: "***split***" and "***merged***"
		
		Supported values for **include_certificate**: (0 and 1) / (***true*** and ***false***) \[Boolean\]"
    
    """
    url = f"https://signeasy5.p.rapidapi.com/v3/rs/embedded/signed/{signed_file_id}/certificate"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signeasy5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_a_pending_envelope(pending_file_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch the details of a specific embedded sending request by passing the pending_file_id in the path parameter."
    
    """
    url = f"https://signeasy5.p.rapidapi.com/v3/rs/embedded/{pending_file_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signeasy5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_all_pending_envelopes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch the list of all the incomplete documents sent for e-signature using embedded sending."
    
    """
    url = f"https://signeasy5.p.rapidapi.com/v3/rs/embedded/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "signeasy5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

