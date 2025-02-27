import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_all_nurses_covid_19(casesns: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return back all the Nurses working on covid-19 from all over the world.
		
		
		Values to add on the required value
		- world
		- america
		- asia
		- africa
		- australia
		- g20
		- asia"
    
    """
    url = f"https://covid-19-data10.p.rapidapi.com/nurses/{casesns}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-data10.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_the_medical_doctors_covid_19(casesdt: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return back all the Medical Doctors working from all over the world in coronavirus.
		
		
		Values to add on the required value
		- world
		- america
		- asia
		- africa
		- australia
		- g20
		- asia"
    
    """
    url = f"https://covid-19-data10.p.rapidapi.com/doctors/{casesdt}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-data10.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_the_icu_beds_covid_19(casesicu: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return back all the ICU Beds available for covid-19 from all over the world.
		
		
		Values to add on the required value
		- world
		- america
		- asia
		- africa
		- australia
		- g20
		- asia"
    
    """
    url = f"https://covid-19-data10.p.rapidapi.com/icu/{casesicu}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-data10.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_the_hospitals_covid_19(casesh: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return back all the hospitals for covid-19 data from all over the world.
		
		
		Values to add on the required value
		- world
		- america
		- asia
		- africa
		- australia
		- g20
		- asia"
    
    """
    url = f"https://covid-19-data10.p.rapidapi.com/hospital/{casesh}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-data10.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_hospital_beds_covid_19(casesbedh: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return back all the Hospital Beds available  from all over the world being used for Covid-19.
		
		
		Values to add on the required value
		- world
		- america
		- asia
		- africa
		- australia
		- g20
		- asia"
    
    """
    url = f"https://covid-19-data10.p.rapidapi.com/hospitalb/{casesbedh}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-data10.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_coronavirus_vaccination_total_covid_19(casesidt: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return back the total number of covid-19 vaccination data from all over the world.
		
		
		Values to add on the required value
		- world
		- america
		- asia
		- africa
		- australia
		- g20
		- asia"
    
    """
    url = f"https://covid-19-data10.p.rapidapi.com/vaccinationt/{casesidt}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-data10.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_coronavirus_vaccination_rate_covid_19(casesdid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return back all the Coronavirus Vaccination Rate  (covid-19) data from all over the world.
		
		Values to add on the required value
		- world
		- america
		- asia
		- africa
		- australia
		- g20
		- asia"
    
    """
    url = f"https://covid-19-data10.p.rapidapi.com/vaccinationr/{casesdid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-data10.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_coronavirus_deaths_covid_19(casesdid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return back all the number of deaths by covid-19 data from all over the world.
		Values to add on the required value
		- world
		- america
		- asia
		- africa
		- australia
		- g20
		- asia"
    
    """
    url = f"https://covid-19-data10.p.rapidapi.com/deathc/{casesdid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-data10.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_coronavirus_cases_covid_19(casesid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return back all the number of cases data from all over the world.
		Parameters to add on the required value
		- world
		- america
		- asia
		- africa
		- australia
		- g20
		- asia"
    
    """
    url = f"https://covid-19-data10.p.rapidapi.com/cases/{casesid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-data10.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

