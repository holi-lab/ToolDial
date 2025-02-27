import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def list_objects(template_id: str='00377b2b1e0ee394', offset: str='0', limit: str='300', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves all the generated PDFs and images
		"
    template_id: Template id
        offset: Offset is used to skip the number of records from the results. Default to 0
        limit: Retrieve only the number of records specified. Default to 300
        
    """
    url = f"https://apitemplateio.p.rapidapi.com/v2/list-objects"
    querystring = {}
    if template_id:
        querystring['template_id'] = template_id
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apitemplateio.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def delete_object(transaction_ref: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Delete a PDF or an image from CDN and mark the transaction as deleted
		"
    transaction_ref: Object transaction reference
        
    """
    url = f"https://apitemplateio.p.rapidapi.com/v2/delete-object"
    querystring = {'transaction_ref': transaction_ref, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apitemplateio.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_templates(limit: str='300', format: str='JPEG', offset: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves the information of templates
		"
    limit: Retrieve only the number of records specified. Default to 300
        format: Either 'PDF' or 'JPEG'
        offset: Offset is used to skip the number of records from the results. Default to 0
        
    """
    url = f"https://apitemplateio.p.rapidapi.com/v2/list-templates"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if format:
        querystring['format'] = format
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apitemplateio.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

