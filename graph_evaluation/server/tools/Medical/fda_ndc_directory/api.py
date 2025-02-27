import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_mdrp_data_by_ndc(ndc: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets drug info about a particular drug in the Medicaid Drug Rebate Program with the national drug code (NDC) identifier."
    
    """
    url = f"https://fda_ndc_directory.p.rapidapi.com/mdrp/product/{ndc}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fda_ndc_directory.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_mdrp_data_by_labeler_code(labeler_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint takes a labeler code and returns all products that labeler (manufacturer) has in the Medicaid Drug Rebate Program."
    
    """
    url = f"https://fda_ndc_directory.p.rapidapi.com/mdrp/labeler/{labeler_code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fda_ndc_directory.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_fda_data_by_ndc(ndc: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets FDA drug info with the national drug code (NDC) identifier.  The backend transforms the NDC parameter supplied and pulls back the FDA info at the the manufacturer-product level (9-digit NDC)."
    
    """
    url = f"https://fda_ndc_directory.p.rapidapi.com/fda/product/{ndc}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fda_ndc_directory.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

