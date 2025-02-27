import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def retrieve_listing_of_processed_emails(timecreateafter: int=None, listname: str='Email list for summer campaign', limit: int=10, timedoneafter: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The listing is sorted by time_done in ascending order. <br><br>Using this method, you can easily retrieve continuously processed emails. To do this, proceed as follows<br>
		    1. Send a request with a minimum limit of 1000 emails.<br>
		    2. Save the retrieved data & newest_time_done in your preferred datastore<br>
		    3. After a few minutes, send another request with timeDoneAfter = saved value newest_time_done.<br><br>
		
		    Notice:
		    The next request may include emails already received in the previous request. This is possible because several emails can finish processing in the same second."
    timecreateafter: Get emails created after UNIX timestamp (operator =>)
        listname: List name
        limit: How many emails to fetch in one request
        timedoneafter: Get emails processed after UNIX timestamp (operator =>)
        
    """
    url = f"https://email-and-list-validation.p.rapidapi.com/verifyEmail/getByStatusDone"
    querystring = {}
    if timecreateafter:
        querystring['timeCreateAfter'] = timecreateafter
    if listname:
        querystring['listName'] = listname
    if limit:
        querystring['limit'] = limit
    if timedoneafter:
        querystring['timeDoneAfter'] = timedoneafter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "email-and-list-validation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reverify_emails_matching_criteria(limit: int=10, listname: str='Email list for summer campaign', timecreateto: int=None, timedoneto: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Emails may become invalid after some time. Use this endpoint to mark chosen emails to be verified again.
		Only emails with processing status DONE are marked for reverification."
    limit: How many emails to set for revalidation. Default limit is 10 as a safety measure. Please change to desired.
        listname: List name
        timecreateto: Reverify emails created before UNIX timestamp (operator <=)
        timedoneto: Reverify emails processed before UNIX timestamp (operator <=)
        
    """
    url = f"https://email-and-list-validation.p.rapidapi.com/verifyEmail/reverify"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if listname:
        querystring['listName'] = listname
    if timecreateto:
        querystring['timeCreateTo'] = timecreateto
    if timedoneto:
        querystring['timeDoneTo'] = timedoneto
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "email-and-list-validation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_summarized_verification_status_for_emails_list(listname: str='Email list for summer campaign', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list overall health."
    listname: List name
        
    """
    url = f"https://email-and-list-validation.p.rapidapi.com/verifyList/getByName"
    querystring = {}
    if listname:
        querystring['listName'] = listname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "email-and-list-validation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_chosen_email_verification_status(email: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get verification status for specific email"
    email: Email address
        
    """
    url = f"https://email-and-list-validation.p.rapidapi.com/verifyEmail/getByEmail"
    querystring = {'email': email, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "email-and-list-validation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def add_single_email_for_verification(email: str, listname: str='Email list for summer campaign', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Add one email for verification."
    email: Email address
        listname: List name
        
    """
    url = f"https://email-and-list-validation.p.rapidapi.com/verifyEmail/single"
    querystring = {'email': email, }
    if listname:
        querystring['listName'] = listname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "email-and-list-validation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

