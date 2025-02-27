import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_datasources_datasource_id_variables(datasource_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This resource list the variables of a specified data source, showing all the fields of the variables."
    
    """
    url = f"https://community-ubidots.p.rapidapi.com/datasources/{datasource_id}/variables"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-ubidots.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_variables_variable_id(variable_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This resource returns a variable and its details. It shows additional fields than the ones required during the creation of the variable:  last_value: Shows the last value written to the variable."
    
    """
    url = f"https://community-ubidots.p.rapidapi.com/variables/{variable_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-ubidots.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_variables(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves the list of all the variables owned by a user. The header X-Auth-Token is needed to have the permissions for this endpoint."
    
    """
    url = f"https://community-ubidots.p.rapidapi.com/variables"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-ubidots.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_datasources(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This resource lists all the existing data sources of a user."
    
    """
    url = f"https://community-ubidots.p.rapidapi.com/datasources"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-ubidots.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_datasources_datasource_id(datasource_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This resource obtains an existing data source and its details. Also, it returns additional fields besides the ones required during the creation of the data source:  number_of_variables: Shows the number of variables contained by the data source. last_activity: Returns the date of the last activity"
    
    """
    url = f"https://community-ubidots.p.rapidapi.com/datasources/{datasource_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-ubidots.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_variables_variable_id_values(variable_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This resource returns the values of the specified variable.  Every value will have a timestamp which is a value in milliseconds according to the POSIX standard."
    
    """
    url = f"https://community-ubidots.p.rapidapi.com/variables/{variable_id}/values"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-ubidots.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

