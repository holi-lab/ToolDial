import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def retrieveclaimlistbyuser(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve Claim List By User"
    
    """
    url = f"https://get-docs.p.rapidapi.com/api/Claim/RetrieveClaimListByUser"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "get-docs.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def case_record(claimid: str='dd61c35e-3edd-ea11-a813-000d3a795762', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "case"
    
    """
    url = f"https://get-docs.p.rapidapi.com/api/Claim/GetDocumentList"
    querystring = {}
    if claimid:
        querystring['claimId'] = claimid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "get-docs.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def document(documentid: str='41aac429-40dd-ea11-a813-000d3a79365a', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "doc"
    
    """
    url = f"https://get-docs.p.rapidapi.com/api/FileManager/GetAuthorizedBlobUriForCourtDocument"
    querystring = {}
    if documentid:
        querystring['documentId'] = documentid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "get-docs.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

