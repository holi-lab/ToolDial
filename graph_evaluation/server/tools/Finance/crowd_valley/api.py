import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def crowd_valley_api_health_status(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check the current health status of the Crowd Valley API."
    
    """
    url = f"https://crowdvalley-crowd-valley-v1.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crowdvalley-crowd-valley-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def check_the_status_of_your_network(network: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check that your network has been set up and is available through the API with your API key."
    
    """
    url = f"https://crowdvalley-crowd-valley-v1.p.rapidapi.com/{network}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crowdvalley-crowd-valley-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def logged_in_user_s_account_information(network: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Populate information on an user's private account page."
    
    """
    url = f"https://crowdvalley-crowd-valley-v1.p.rapidapi.com/{network}/self"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crowdvalley-crowd-valley-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_the_user_s_investments(network: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a list of the user's investments. This will include open investment offers as well as closed investments."
    
    """
    url = f"https://crowdvalley-crowd-valley-v1.p.rapidapi.com/{network}/self/investments"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crowdvalley-crowd-valley-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def the_user_s_organizations(network: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of the logged-in User's Organizations. This could be used to show for which Organization(s) the User works or for which Organization(s) the User has created Offerings."
    
    """
    url = f"https://crowdvalley-crowd-valley-v1.p.rapidapi.com/{network}/self/organizations"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crowdvalley-crowd-valley-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def visible_users_in_your_network(network: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Populate a public listing page of your visible Users. Users are visible if they have their Visibility field set to Open."
    
    """
    url = f"https://crowdvalley-crowd-valley-v1.p.rapidapi.com/{network}/users"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crowdvalley-crowd-valley-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_a_list_of_organizations(network: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of visible Organizations in the network. Implement this to show a list of Organizations that are supporting various deals, or that are umbrella entities for various users. This will only show Organizations that have visibility set to Open."
    
    """
    url = f"https://crowdvalley-crowd-valley-v1.p.rapidapi.com/{network}/organizations"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crowdvalley-crowd-valley-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_information_about_this_organization(network: str, organizationid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Information about the Organization. This will only work if the logged-in User is a team member of the Organization or the Organization has visibility set to Open."
    
    """
    url = f"https://crowdvalley-crowd-valley-v1.p.rapidapi.com/{network}/organizations/{organizationid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crowdvalley-crowd-valley-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_offerings_for_this_organization(network: str, organizationid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of visible Offerings for this Organization. Visible Offerings are those with status Restricted, Published, Live, Closing or Settled."
    
    """
    url = f"https://crowdvalley-crowd-valley-v1.p.rapidapi.com/{network}/organizations/{organizationid}/offerings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crowdvalley-crowd-valley-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_offerings(network: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Populate a live Offerings page or a historical/closed deals page. Visible Offerings are those with status Restricted, Published, Live, Closing or Settled."
    
    """
    url = f"https://crowdvalley-crowd-valley-v1.p.rapidapi.com/{network}/offerings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crowdvalley-crowd-valley-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def information_about_this_offering(network: str, offeringid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Information about this Offering. This will only work if the Offering is visible in the network."
    
    """
    url = f"https://crowdvalley-crowd-valley-v1.p.rapidapi.com/{network}/offerings/{offeringid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crowdvalley-crowd-valley-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_investments_for_this_offering(network: str, offeringid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Populate a list of Investments for this Offering. This can be used to show a list of current investors for this Offering. Investments will be shown in full if visibility is set to Open."
    
    """
    url = f"https://crowdvalley-crowd-valley-v1.p.rapidapi.com/{network}/offerings/{offeringid}/investments"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crowdvalley-crowd-valley-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def information_about_this_investment(network: str, investmentid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Information about this Investment. It will only work if the Investment has visibility set to Open or Anonymous."
    
    """
    url = f"https://crowdvalley-crowd-valley-v1.p.rapidapi.com/{network}/investments/{investmentid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crowdvalley-crowd-valley-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_a_user_s_public_profile_information(network: str, userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Populate a user's public profile information. Implement this to allow the logged-in user to see information about other users on the network. It will only populate all information if the target user's visibility is set to Open. This function will only show public profile information - it will not show private account information."
    
    """
    url = f"https://crowdvalley-crowd-valley-v1.p.rapidapi.com/{network}/users/{userid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crowdvalley-crowd-valley-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

