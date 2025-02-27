import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_id_by_name(companyname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given a company's name string as input, returns crunchbase's unque id for that company. That unique id can then be used to pull additional information about the company."
    
    """
    url = f"https://crunchbase-company-data1.p.rapidapi.com/getCbIdByName"
    querystring = {'companyName': companyname, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crunchbase-company-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_data_by_name(companyname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given a company's name string, the endpoint returns all available Crunchbase data on this company."
    
    """
    url = f"https://crunchbase-company-data1.p.rapidapi.com/getAllCbDataByName"
    querystring = {'companyName': companyname, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crunchbase-company-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_investors_data_by_name(companyname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given a company's name string as input, returns a real-time scrape of all available investor data from Crunchbase"
    
    """
    url = f"https://crunchbase-company-data1.p.rapidapi.com/getCbInvestorsDataByName"
    querystring = {'companyName': companyname, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crunchbase-company-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_id_by_domain(domain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given a company's domain as input, returns CrunchBase's unique id for that company. Be aware, can provide erroneous results with Crunchbase has two entries for the same company. For example, plugging in snapchat.com returns the incorrect id for Snapchat whereas plugging in snap.com returns the correct info."
    
    """
    url = f"https://crunchbase-company-data1.p.rapidapi.com/getCbIdByDomain"
    querystring = {'domain': domain, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crunchbase-company-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_cb_data_by_id(thisuuid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given a company's id, the endpoint returns all availalbe crunchbase data for this company."
    
    """
    url = f"https://crunchbase-company-data1.p.rapidapi.com/getAllCbDataById"
    querystring = {'thisUuid': thisuuid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crunchbase-company-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_funding_data_by_name(companyname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given a company name string, returns all available funding data on that company scraped from crunchbase in real time."
    
    """
    url = f"https://crunchbase-company-data1.p.rapidapi.com/getCbFundingDataByName"
    querystring = {'companyName': companyname, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crunchbase-company-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_acquisitions_data_by_id(thisuuid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given a company's id as input, returns a real-time scrape of all available acquisition data from Crunchbase - i.e. all the acquisitions made by the company id provided"
    
    """
    url = f"https://crunchbase-company-data1.p.rapidapi.com/getCbAcquisitionsDataById"
    querystring = {'thisUuid': thisuuid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crunchbase-company-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_people_data_by_id(thisuuid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given a company's unique id as input, returns a real-time scrape of all available people data from Crunchbase."
    
    """
    url = f"https://crunchbase-company-data1.p.rapidapi.com/getCbPeopleDataById"
    querystring = {'thisUuid': thisuuid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crunchbase-company-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_funding_data_by_id(thisuuid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given a company's unique id as input, returns all available funding data from a real-time scrape of Crunchbase."
    
    """
    url = f"https://crunchbase-company-data1.p.rapidapi.com/getCbFundingDataById"
    querystring = {'thisUuid': thisuuid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crunchbase-company-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_investors_data_by_id(thisuuid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given a company's Crunchbase id, returns a real-time scrape of all available investor data from Crunchbase."
    
    """
    url = f"https://crunchbase-company-data1.p.rapidapi.com/getCbInvestorsDataById"
    querystring = {'thisUuid': thisuuid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crunchbase-company-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_acquisitions_data_by_name(companyname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given a company's name string as input, returns a real-time scrape of all available acquisitions data from Crunchbase"
    
    """
    url = f"https://crunchbase-company-data1.p.rapidapi.com/getCbAcquisitionsDataByName"
    querystring = {'companyName': companyname, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crunchbase-company-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_people_data_by_name(companyname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given a company's name string as input, returns all available data about senior people at the company from crunchbase."
    
    """
    url = f"https://crunchbase-company-data1.p.rapidapi.com/getCbPeopleDataByName"
    querystring = {'companyName': companyname, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crunchbase-company-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

