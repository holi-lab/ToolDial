import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def download_verification_results(filename: str, authorization: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the result file to be downloaded.
		
		Pass name of file to be downloaded as path param. Result file name is always in format {filename}_output.xlsx Replace {filename} with the name returned from upload file call.
		
		To download the file with angular follow sample code here
		
		https://stackoverflow.com/questions/40240796/angular-2-best-approach-to-use-filesaver-js"
    filename: Name of file to be downloaded. Result file name is always in format {filename}_output.xlsx Replace {filename} with the name returned from upload file call.
        authorization: Bearer <access_token>. You get the "access_token" as response of login. Use it in header.
        
    """
    url = f"https://phone-verification3.p.rapidapi.com/api/v1/file/downloadFile/{filename}"
    querystring = {}
    if authorization:
        querystring['Authorization'] = authorization
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "phone-verification3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def verify_single_number(authorization: str, number: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details of phone number supplied in the query param. Do not forget to pass Authorization header with your access_token. 
		
		Hint - access_token are returned as part of login api call."
    authorization: Bearer <access_token>. You get the "access_token" as response of login. Use it in header.
        number: Phone/mobile number to get details for.
        
    """
    url = f"https://phone-verification3.p.rapidapi.com/api/v1/validation/verifyNumber"
    querystring = {'Authorization': authorization, 'number': number, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "phone-verification3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_processing_status_of_file(authorization: str, filename: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives you status of file uploaded for verification."
    authorization: Bearer <access_token>. You get the "access_token" as response of login. Use it in header.
        filename: file name received in response of upload file call. 
        
    """
    url = f"https://phone-verification3.p.rapidapi.com/api/v1/documentProcess/getProcessingStatus"
    querystring = {'Authorization': authorization, 'fileName': filename, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "phone-verification3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

