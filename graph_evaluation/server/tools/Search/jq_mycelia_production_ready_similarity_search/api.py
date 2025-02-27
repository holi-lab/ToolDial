import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_index_index_db_name_get(db_name: str, mode: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get index info of a given databse.
		
		Parameters
		----------
		db_name : str
		    Name of the database we want index info from.
		
		mode : Mode, optional
		    Response mode of indexes. We have three options:
		    - complete: get ALL indexes.
		    - summarized: get groups of consecutive indexes.
		    - simple: get the number of indexes and their min and max indexes as well.
		
		Returns
		-------
		List
		    Returns a list of integers or strings depending on the mode parameter."
    db_name: Database name
        mode: Detail of response expected.
        
    """
    url = f"https://jq-mycelia-production-ready-similarity-search.p.rapidapi.com/index/{db_name}"
    querystring = {}
    if mode:
        querystring['mode'] = mode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jq-mycelia-production-ready-similarity-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_table_fields_table_fields_db_name_get(db_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the column names of a SUPERVISED/UNSUPERVISED database.
		
		Parameters
		----------
		db_name : str
		    Name of database to get the column names from.
		
		Returns
		-------
		Dict
		    Dictionary with the name of columns as keys and an example of their values as... values.
		    int -> 0
		    float -> 0.01
		    string -> "string""
    db_name: Database name
        
    """
    url = f"https://jq-mycelia-production-ready-similarity-search.p.rapidapi.com/table/fields/{db_name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jq-mycelia-production-ready-similarity-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_status_status_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get status of API tasks.
		So far, provides status of background tasks (202).
		
		Returns
		-------
		last_status : Dict
		    {'Task': 'string',
		     'Status': 'string',
		     'Description': 'string'}."
    
    """
    url = f"https://jq-mycelia-production-ready-similarity-search.p.rapidapi.com/status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jq-mycelia-production-ready-similarity-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_vector_vector_db_name_get(db_name: str, index: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get vectors according to the specified indexes.
		
		Parameters
		----------
		db_name : string
		    Database we want to extract vectors from.
		
		index : list
		    List of indexes associated with the vectors we are interested in.
		
		Returns
		-------
		response : list of dict
		    List of dictionaries. Each dict entry has an index and its corresponding vector."
    db_name: Database name
        index: The ID of the item to return the vector
        
    """
    url = f"https://jq-mycelia-production-ready-similarity-search.p.rapidapi.com/vector/{db_name}"
    querystring = {'index': index, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jq-mycelia-production-ready-similarity-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def check_inserted_data_table_setup_index_db_name_get(db_name: str, mode: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the indexes of data inserted with /table/setup/data
		
		Parameters
		---------------
		db_name : str
		    Name of database to check the inserted data.
		
		mode : Mode, optional
		    Response mode of indexes. We have three options:
		    - complete: get ALL indexes.
		    - summarized: get groups of consecutive indexes.
		    - simple: get the number of indexes and their min and max indexes as well.
		
		Returns
		-------
		List
		    Returns a list of integers or strings depending on the mode parameter."
    db_name: Database name
        mode: Detail of response expected.
        
    """
    url = f"https://jq-mycelia-production-ready-similarity-search.p.rapidapi.com/table/setup/index/{db_name}"
    querystring = {}
    if mode:
        querystring['mode'] = mode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jq-mycelia-production-ready-similarity-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def check_valid_name_valid_db_name_get(db_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check if a database name is valid.
		
		Returns
		-------
		response : Whether or not the database name is valid."
    db_name: Database name
        
    """
    url = f"https://jq-mycelia-production-ready-similarity-search.p.rapidapi.com/valid/{db_name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jq-mycelia-production-ready-similarity-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_database_names_names_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the names of each database in your config file.
		
		Returns
		-------
		response : List of databases in your config file."
    
    """
    url = f"https://jq-mycelia-production-ready-similarity-search.p.rapidapi.com/names"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jq-mycelia-production-ready-similarity-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_database_info_dev_info_db_name_get(db_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a detailed config of a specified database. This is a dev method.
		
		Parameters
		----------
		db_name : str
		    Database name we want the detailed config.
		
		Returns
		-------
		response : Dict
		    Dictionary with the parameters of the specified database."
    db_name: Database name
        
    """
    url = f"https://jq-mycelia-production-ready-similarity-search.p.rapidapi.com/dev/info/{db_name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jq-mycelia-production-ready-similarity-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

