import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_drug_info_by_drug_name_route(drgrte: str, drgnm: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This web method will return all the dataset related to the drug information using the “drgNm” the drug name or brand name or drug product and “drgRte” drug route, these input parameters are required.  All web method requires the "api_key" as query parameter."
    
    """
    url = f"https://konviere-drugsapi.p.rapidapi.com/konviere/drugs/getDrugInfoByDrgNm&DrgRte/{drgnm}/{drgrte}"
    querystring = {'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "konviere-drugsapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_adverse_event_info_by_reaction_sex_age_group_drug_name(drgnm: str, sex: int, reaction: str, api_key: str, agegrp: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This web method will return a summary dataset related to the adverse events information using the “reaction” the reaction, “sex” the patient sex (0: unknown, 1: male, 2: female), “ageGrp” age group (1: Neonate, 2: Infant,3: Child,4: Adolescent,5: Adult,6: Elderly) and “drgNm” the drug name or brand 
		name, these input parameters are required.  All web method requires the "api_key" as query parameter."
    
    """
    url = f"https://konviere-drugsapi.p.rapidapi.com/Konviere/drugs/getReactionsByReaction&Sex&AgeGrp&DrgNm/{reaction}/{sex}/{agegrp}/{drgnm}"
    querystring = {'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "konviere-drugsapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_adverse_event_info_by_reaction_sex_age_group(agegrp: int, sex: int, api_key: str, reaction: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This web method will return a summary dataset related to the adverse events information using the “reaction” the reaction, “sex” the patient sex (0: unknown, 1: male, 2: female) and “ageGrp” age group (1: Neonate, 2: Infant,3: Child,4: Adolescent,5: Adult,6: Elderly), these input parameters are required.  All web method requires the "api_key" as query parameter."
    
    """
    url = f"https://konviere-drugsapi.p.rapidapi.com/Konviere/drugs/getReactionsByReaction&Sex&AgeGrp/{reaction}/{sex}/{agegrp}"
    querystring = {'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "konviere-drugsapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_adverse_event_info_by_reaction(reaction: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This web method will return a summary dataset related to the adverse events information using the “reaction” the reaction that was reported, this input parameter is required.  All web method requires the "api_key" as query parameter."
    
    """
    url = f"https://konviere-drugsapi.p.rapidapi.com/konviere/drugs/getReactionsByReaction/{reaction}"
    querystring = {'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "konviere-drugsapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_adverse_event_info_by_reaction_sex(sex: int, api_key: str, reaction: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This web method will return a summary dataset related to the adverse events information using the “reaction” the reaction and “sex” the patient sex (0: unknown, 1: male, 2: female), these input parameters are required.
		All web method requires the "api_key" as query parameter."
    
    """
    url = f"https://konviere-drugsapi.p.rapidapi.com/Konviere/drugs/getReactionsByReaction&Sex/{reaction}/{sex}"
    querystring = {'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "konviere-drugsapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_adverse_event_summary_info_by_pharmacologic_class(pharmacologicclass: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This web method will return a summary dataset related to the adverse events information using the “pharmacologicClass” the pharmacologic class, this input parameter is required.  All web method requires the "api_key" as query parameter."
    
    """
    url = f"https://konviere-drugsapi.p.rapidapi.com/konviere/drugs/getAdverseEventsSummaryByPharmClass/{pharmacologicclass}"
    querystring = {'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "konviere-drugsapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

