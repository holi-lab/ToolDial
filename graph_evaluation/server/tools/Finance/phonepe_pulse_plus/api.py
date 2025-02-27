import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_top_states_districts_pincodes_for_transactions(quarter: str, year: int, state: str='maharashtra', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Top States/Districts/Pincodes with transactions data."
    state: Specify State to narrow down results to a particular state.

Allowed values -

andaman-&-nicobar-islands
andhra-pradesh
arunachal-pradesh
assam
bihar
chandigarh
chhattisgarh
dadra-&-nagar-haveli-&-daman-&-diu
delhi
goa
gujarat
haryana
himachal-pradesh
jammu-&-kashmir
jharkhand
karnataka
kerala
ladakh
lakshadweep
madhya-pradesh
maharashtra
manipur
meghalaya
mizoram
nagaland
odisha
puducherry
punjab
rajasthan
sikkim
tamil-nadu
telangana
tripura
uttar-pradesh
uttarakhand
west-bengal
        
    """
    url = f"https://phonepe-pulse-plus.p.rapidapi.com/top/transaction/{year}/{quarter}"
    querystring = {}
    if state:
        querystring['state'] = state
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "phonepe-pulse-plus.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_aggregated_users_data(quarter: str, year: int, state: str='maharashtra', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Aggregated users data across all states for a specific quarter in a year"
    state: State to narrow down results to a particular state.

Allowed values -

andaman-&-nicobar-islands
andhra-pradesh
arunachal-pradesh
assam
bihar
chandigarh
chhattisgarh
dadra-&-nagar-haveli-&-daman-&-diu
delhi
goa
gujarat
haryana
himachal-pradesh
jammu-&-kashmir
jharkhand
karnataka
kerala
ladakh
lakshadweep
madhya-pradesh
maharashtra
manipur
meghalaya
mizoram
nagaland
odisha
puducherry
punjab
rajasthan
sikkim
tamil-nadu
telangana
tripura
uttar-pradesh
uttarakhand
west-bengal
        
    """
    url = f"https://phonepe-pulse-plus.p.rapidapi.com/aggregated/user/{year}/{quarter}"
    querystring = {}
    if state:
        querystring['state'] = state
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "phonepe-pulse-plus.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_top_states_districts_pincodes_for_users(quarter: str, year: int, state: str='Maharashtra', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Top States/Districts/Pincodes with users data."
    state: Specify State to narrow down results to a particular state.

Allowed values -

andaman-&-nicobar-islands
andhra-pradesh
arunachal-pradesh
assam
bihar
chandigarh
chhattisgarh
dadra-&-nagar-haveli-&-daman-&-diu
delhi
goa
gujarat
haryana
himachal-pradesh
jammu-&-kashmir
jharkhand
karnataka
kerala
ladakh
lakshadweep
madhya-pradesh
maharashtra
manipur
meghalaya
mizoram
nagaland
odisha
puducherry
punjab
rajasthan
sikkim
tamil-nadu
telangana
tripura
uttar-pradesh
uttarakhand
west-bengal
        
    """
    url = f"https://phonepe-pulse-plus.p.rapidapi.com/top/user/{year}/{quarter}"
    querystring = {}
    if state:
        querystring['state'] = state
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "phonepe-pulse-plus.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_district_wise_users_data(quarter: str, year: int, state: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "District-wise users data for a particular state for a specific quarter in a year"
    state: Allowed values -

andaman-&-nicobar-islands
andhra-pradesh
arunachal-pradesh
assam
bihar
chandigarh
chhattisgarh
dadra-&-nagar-haveli-&-daman-&-diu
delhi
goa
gujarat
haryana
himachal-pradesh
jammu-&-kashmir
jharkhand
karnataka
kerala
ladakh
lakshadweep
madhya-pradesh
maharashtra
manipur
meghalaya
mizoram
nagaland
odisha
puducherry
punjab
rajasthan
sikkim
tamil-nadu
telangana
tripura
uttar-pradesh
uttarakhand
west-bengal
        
    """
    url = f"https://phonepe-pulse-plus.p.rapidapi.com/district-wise/user/{state}/{year}/{quarter}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "phonepe-pulse-plus.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_aggregated_transaction_data(year: int, quarter: str, state: str='maharashtra', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Aggregated transaction data across all states for a specific quarter in a year"
    state: Specify State to narrow down results to a particular state.

Allowed values -

andaman-&-nicobar-islands
andhra-pradesh
arunachal-pradesh
assam
bihar
chandigarh
chhattisgarh
dadra-&-nagar-haveli-&-daman-&-diu
delhi
goa
gujarat
haryana
himachal-pradesh
jammu-&-kashmir
jharkhand
karnataka
kerala
ladakh
lakshadweep
madhya-pradesh
maharashtra
manipur
meghalaya
mizoram
nagaland
odisha
puducherry
punjab
rajasthan
sikkim
tamil-nadu
telangana
tripura
uttar-pradesh
uttarakhand
west-bengal
        
    """
    url = f"https://phonepe-pulse-plus.p.rapidapi.com/aggregated/transaction/{year}/{quarter}"
    querystring = {}
    if state:
        querystring['state'] = state
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "phonepe-pulse-plus.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_district_wise_transaction_data(year: int, quarter: str, state: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "District-wise transaction data for a particular state for a specific quarter in a year"
    state: Allowed values -

andaman-&-nicobar-islands
andhra-pradesh
arunachal-pradesh
assam
bihar
chandigarh
chhattisgarh
dadra-&-nagar-haveli-&-daman-&-diu
delhi
goa
gujarat
haryana
himachal-pradesh
jammu-&-kashmir
jharkhand
karnataka
kerala
ladakh
lakshadweep
madhya-pradesh
maharashtra
manipur
meghalaya
mizoram
nagaland
odisha
puducherry
punjab
rajasthan
sikkim
tamil-nadu
telangana
tripura
uttar-pradesh
uttarakhand
west-bengal
        
    """
    url = f"https://phonepe-pulse-plus.p.rapidapi.com/district-wise/transaction/{state}/{year}/{quarter}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "phonepe-pulse-plus.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_state_wise_users_data(quarter: str, year: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "State-wise users data across all states for a specific quarter in a year"
    
    """
    url = f"https://phonepe-pulse-plus.p.rapidapi.com/state-wise/user/{year}/{quarter}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "phonepe-pulse-plus.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_state_wise_transaction_data(year: int, quarter: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "State-wise transaction data across all states for a specific quarter in a year"
    
    """
    url = f"https://phonepe-pulse-plus.p.rapidapi.com/state-wise/transaction/{year}/{quarter}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "phonepe-pulse-plus.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

