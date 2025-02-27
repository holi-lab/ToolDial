import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_cci_by_number(cci: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return a single CCI and it's definition based on the 6-digit CCI identification number."
    
    """
    url = f"https://risk-management-framework.p.rapidapi.com/getCci/{cci}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "risk-management-framework.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_subcontrol_by_control(control: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return all the subcontrols/ccis owned by a control such as AC-1. In the case of AC-1, this endpoint will return AC-1.1, AC-1.2, etc."
    
    """
    url = f"https://risk-management-framework.p.rapidapi.com/getSubcontrolByControl/{control}"
    querystring = {}
    if control:
        querystring['control'] = control
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "risk-management-framework.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_controls(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a complete list of all Risk Management Framework (RMF) controls."
    
    """
    url = f"https://risk-management-framework.p.rapidapi.com/getAllControls"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "risk-management-framework.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_subcontrols(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all the subcontrols of all main controls. 
		
		Ex/ AC-1 is a main control while AC-1.1 is a subcontrol."
    
    """
    url = f"https://risk-management-framework.p.rapidapi.com/getAllSubControls"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "risk-management-framework.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_ccis_aps(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all Control Correlation Identifiers (CCIs)  AKA Assessment Procedures (APs)."
    
    """
    url = f"https://risk-management-framework.p.rapidapi.com/getAllCcis"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "risk-management-framework.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

