import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def read_a_survey_nlp(sid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a detail of customer survey answer by its survey id (sid), and applies to the third answer (a3) the sentiment analysis feature."
    
    """
    url = f"https://nps-net-promoter-score.p.rapidapi.com/nps/survey/read/nlp/{sid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nps-net-promoter-score.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nps_organization(oid: str, start_date: str, end_date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a summary of NPS calculation for a given organization. It considers the overall answers related to all templates belonging to a given organization and consolidates in a global NPS indicator. A time period is mandatory (start_date and  end_date) to narrow the results, otherwise, it will consider the last 24hs."
    
    """
    url = f"https://nps-net-promoter-score.p.rapidapi.com/nps/report/organization/{oid}"
    querystring = {'start_date': start_date, 'end_date': end_date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nps-net-promoter-score.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def details_all_templates_answers(tid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a full list of all customer survey responses for a given template, by identifying its template id (tid)"
    
    """
    url = f"https://nps-net-promoter-score.p.rapidapi.com/nps/report/surveys/template/{tid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nps-net-promoter-score.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def details_all_organization_surveys(oid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a full list of all customer survey responses for a given organization, by identifying its organization id (oid)"
    
    """
    url = f"https://nps-net-promoter-score.p.rapidapi.com/nps/report/surveys/organization/{oid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nps-net-promoter-score.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nps_template(tid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a summary of NPS calculation for a given template. It considers the overall answers related to that template and consolidates in a global NPS indicator. A time period is mandatory (start_date and  end_date) to narrow the results, otherwise, it will consider the last 24hs."
    
    """
    url = f"https://nps-net-promoter-score.p.rapidapi.com/nps/report/template/{tid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nps-net-promoter-score.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def read_all_templates_surveys(tid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of all customer surveys answers related to a given template of questions (tid). In this method, you can see how each customer answered a given template of questions"
    
    """
    url = f"https://nps-net-promoter-score.p.rapidapi.com/nps/survey/read/template/{tid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nps-net-promoter-score.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_all_nps_organization_templates(oid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of all NPS templates of a given Organization by its organization id (oid)"
    
    """
    url = f"https://nps-net-promoter-score.p.rapidapi.com/nps/template/list/organization/{oid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nps-net-promoter-score.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def read_a_survey(sid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details of a given customer survey by its survey id (sid)"
    
    """
    url = f"https://nps-net-promoter-score.p.rapidapi.com/nps/survey/read/{sid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nps-net-promoter-score.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def read_organization(oid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a organization record by its organization id (oid)"
    
    """
    url = f"https://nps-net-promoter-score.p.rapidapi.com/nps/organization/read/{oid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nps-net-promoter-score.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nps_client(cid: str, start_date: str, end_date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a summary of NPS calculation for a given client_id. It considers the overall answers related to all survey answers belonging to a given client and consolidates in a global NPS indicator. A time period is mandatory (start_date and  end_date) to narrow the results, otherwise, it will consider the last 24hs."
    
    """
    url = f"https://nps-net-promoter-score.p.rapidapi.com/nps/report/client/{cid}"
    querystring = {'start_date': start_date, 'end_date': end_date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nps-net-promoter-score.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def read_nps_template(tid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get details of a given template by its template id (tid)"
    
    """
    url = f"https://nps-net-promoter-score.p.rapidapi.com/nps/template/read/{tid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nps-net-promoter-score.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

