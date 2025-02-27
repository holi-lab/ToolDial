import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_time_series_data_and_metadata(database_code: str, return_format: str, dataset_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This call returns data and metadata for a given time-series."
    database_code: The code for the database this time-series belongs to
        return_format: json, xml or csv
        dataset_code: The code for this time-series
        
    """
    url = f"https://quandl1.p.rapidapi.com/datasets/{database_code}/{dataset_code}.{return_format}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quandl1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_filtered_time_series_data(dataset_code: str, return_format: str, database_code: str, column_index: str='4', end_date: str='2014-12-31', limit: str=None, start_date: str='2014-01-01', collapse: str='monthly', order: str='desc', transform: str='rdiff', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can slice, transform and otherwise customize your time-series dataset prior to download by appending various optional parameters to your query.  Get monthly % changes in Facebook's closing price for the year 2014:"
    dataset_code: The code for this time-series
        return_format: json, xml or csv
        database_code: The code for the database this time-series belongs to
        column_index: Request a specific column. Column 0 is the date column and is always returned. Data begins at column 1.
        end_date: Retrieve data rows up to and including the specified end date.
        limit: Use limit=n to get the first n rows of the dataset. Use limit=1 to get just the latest row.
        start_date: Retrieve data rows on and after the specified start date.
        collapse: Change the sampling frequency of the returned data. Default is none; i.e., data is returned in its original granularity.
        order: Return data in ascending or descending order of date. Default is desc.
        transform: Perform elementary calculations on the data prior to downloading. Default is none. Calculation options are described below.
        
    """
    url = f"https://quandl1.p.rapidapi.com/datasets/{database_code}/{dataset_code}.{return_format}"
    querystring = {}
    if column_index:
        querystring['column_index'] = column_index
    if end_date:
        querystring['end_date'] = end_date
    if limit:
        querystring['limit'] = limit
    if start_date:
        querystring['start_date'] = start_date
    if collapse:
        querystring['collapse'] = collapse
    if order:
        querystring['order'] = order
    if transform:
        querystring['transform'] = transform
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quandl1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_metadata_for_a_time_series_database(database_code: str, return_format: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can retrieve metadata for a specified time-series database"
    database_code: The code for the database this time-series belongs to
        return_format: json, xml or csv
        
    """
    url = f"https://quandl1.p.rapidapi.com/databases/{database_code}.{return_format}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quandl1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_time_series_data(return_format: str, dataset_code: str, database_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This call returns data from a specified time-series."
    return_format: How you want the data returned (json, xml, or csv)
        dataset_code: The code for this time-series
        database_code: The code for the database this time-series belongs to
        
    """
    url = f"https://quandl1.p.rapidapi.com/datasets/{database_code}/{dataset_code}/data.{return_format}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quandl1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_time_series_metadata(return_format: str, database_code: str, dataset_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This call returns metadata for a specified time-series."
    return_format: json, xml or csv
        database_code: The code for the database this time-series belongs to
        dataset_code: The code for this time-series
        
    """
    url = f"https://quandl1.p.rapidapi.com/datasets/{database_code}/{dataset_code}/metadata.{return_format}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quandl1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

