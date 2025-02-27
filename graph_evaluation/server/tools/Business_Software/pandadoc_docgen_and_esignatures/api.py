import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def listapilogs(authorization1: str, authorization: str='Bearer {{access_token}}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pandadoc-docgen-and-esignatures.p.rapidapi.com/public/v1/logs"
    querystring = {'Authorization1': authorization1, }
    if authorization:
        querystring['Authorization'] = authorization
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pandadoc-docgen-and-esignatures.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def downloaddocument(authorization1: str, document_id: str, authorization: str='Bearer {{access_token}}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For details go to [https://developers.pandadoc.com/v1/reference#download-document](https://developers.pandadoc.com/v1/reference#download-document)."
    
    """
    url = f"https://pandadoc-docgen-and-esignatures.p.rapidapi.com/public/v1/documents/{document_id}/download"
    querystring = {'Authorization1': authorization1, }
    if authorization:
        querystring['Authorization'] = authorization
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pandadoc-docgen-and-esignatures.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getaccesscode(scope: str, client_id: str, response_type: str, redirect_uri: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide values to variables: client_id, redirect_url ([Postman environments](https://www.getpostman.com/docs/postman/environments_and_globals/manage_environments) is great for it).
		
		Test this link in a web browser.
		
		You can get the URL by going to "Code" - "cURL".
		
		For details go to [https://developers.pandadoc.com/v1/reference#authorize-a-user](https://developers.pandadoc.com/v1/reference#authorize-a-user).
		
		For details of authentication process go to [https://developers.pandadoc.com/v1/reference#authentication-process](https://developers.pandadoc.com/v1/reference#authentication-process)."
    
    """
    url = f"https://pandadoc-docgen-and-esignatures.p.rapidapi.com/oauth2/authorize"
    querystring = {'scope': scope, 'client_id': client_id, 'response_type': response_type, 'redirect_uri': redirect_uri, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pandadoc-docgen-and-esignatures.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listtemplates(authorization1: str, authorization: str='Bearer {{access_token}}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For details go to [https://developers.pandadoc.com/v1/reference#list-templates](https://developers.pandadoc.com/v1/reference#list-templates)."
    
    """
    url = f"https://pandadoc-docgen-and-esignatures.p.rapidapi.com/public/v1/templates"
    querystring = {'Authorization1': authorization1, }
    if authorization:
        querystring['Authorization'] = authorization
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pandadoc-docgen-and-esignatures.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listdocuments(authorization1: str, authorization: str='Bearer {{access_token}}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For details go to [https://developers.pandadoc.com/v1/reference#list-documents](https://developers.pandadoc.com/v1/reference#list-documents)."
    
    """
    url = f"https://pandadoc-docgen-and-esignatures.p.rapidapi.com/public/v1/documents"
    querystring = {'Authorization1': authorization1, }
    if authorization:
        querystring['Authorization'] = authorization
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pandadoc-docgen-and-esignatures.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def documentdetails(authorization1: str, document_id: str, authorization: str='Bearer {{access_token}}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For details go to [https://developers.pandadoc.com/v1/reference#document-details](https://developers.pandadoc.com/v1/reference#document-details)."
    
    """
    url = f"https://pandadoc-docgen-and-esignatures.p.rapidapi.com/public/v1/documents/{document_id}/details"
    querystring = {'Authorization1': authorization1, }
    if authorization:
        querystring['Authorization'] = authorization
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pandadoc-docgen-and-esignatures.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def contentlibraryitemdetails(authorization1: str, cli_id: str, authorization: str='Bearer {{access_token}}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pandadoc-docgen-and-esignatures.p.rapidapi.com/public/v1/content-library-items/{cli_id}/details"
    querystring = {'Authorization1': authorization1, }
    if authorization:
        querystring['Authorization'] = authorization
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pandadoc-docgen-and-esignatures.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listdocumentsfolders(authorization1: str, authorization: str='Bearer {{access_token}}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For details go to https://developers.pandadoc.com/reference#list-documents-folders"
    
    """
    url = f"https://pandadoc-docgen-and-esignatures.p.rapidapi.com/public/v1/documents/folders"
    querystring = {'Authorization1': authorization1, }
    if authorization:
        querystring['Authorization'] = authorization
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pandadoc-docgen-and-esignatures.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def templatedetails(authorization1: str, template_id: str, authorization: str='Bearer {{access_token}}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For details go to [https://developers.pandadoc.com/v1/reference#template-details](https://developers.pandadoc.com/v1/reference#template-details)."
    
    """
    url = f"https://pandadoc-docgen-and-esignatures.p.rapidapi.com/public/v1/templates/{template_id}/details"
    querystring = {'Authorization1': authorization1, }
    if authorization:
        querystring['Authorization'] = authorization
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pandadoc-docgen-and-esignatures.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listcontentlibraryitems(authorization1: str, authorization: str='Bearer {{access_token}}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pandadoc-docgen-and-esignatures.p.rapidapi.com/public/v1/content-library-items"
    querystring = {'Authorization1': authorization1, }
    if authorization:
        querystring['Authorization'] = authorization
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pandadoc-docgen-and-esignatures.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def documentstatus(authorization1: str, document_id: str, authorization: str='Bearer {{access_token}}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For details go to [https://developers.pandadoc.com/v1/reference#document-status](https://developers.pandadoc.com/v1/reference#document-status)."
    
    """
    url = f"https://pandadoc-docgen-and-esignatures.p.rapidapi.com/public/v1/documents/{document_id}"
    querystring = {'Authorization1': authorization1, }
    if authorization:
        querystring['Authorization'] = authorization
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pandadoc-docgen-and-esignatures.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getapikeyathttp_app_pandadoc_com_a_settings_integrations_api(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pandadoc-docgen-and-esignatures.p.rapidapi.com/a/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pandadoc-docgen-and-esignatures.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def testsharablelink(session_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Test this link in a web browser.
		
		You can get the URL by going to "Code" - "cURL".
		
		For details go to [https://developers.pandadoc.com/v1/reference#link-to-a-document](https://developers.pandadoc.com/v1/reference#link-to-a-document)."
    
    """
    url = f"https://pandadoc-docgen-and-esignatures.p.rapidapi.com/s/{session_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pandadoc-docgen-and-esignatures.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def apilogdetails(authorization1: str, eventid: str, authorization: str='Bearer {{access_token}}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pandadoc-docgen-and-esignatures.p.rapidapi.com/public/v1/logs/{eventid}"
    querystring = {'Authorization1': authorization1, }
    if authorization:
        querystring['Authorization'] = authorization
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pandadoc-docgen-and-esignatures.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listtemplatesfolders(authorization1: str, authorization: str='Bearer {{access_token}}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For details go to https://developers.pandadoc.com/reference#list-templates-folders"
    
    """
    url = f"https://pandadoc-docgen-and-esignatures.p.rapidapi.com/public/v1/templates/folders"
    querystring = {'Authorization1': authorization1, }
    if authorization:
        querystring['Authorization'] = authorization
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pandadoc-docgen-and-esignatures.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

