import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def remove_addresses(message: str, greedy: bool=None, replacementtext: str='<address replaced>', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Remove all addresses from a given string of text. This will match short and longhand address such as: 111 Main St. or 111 Main Street, San Francisco, CA 94107"
    message: This is a string which will be stripped of addresses
        greedy: Setting greedy to true will cause the replacement to be more likely to match false positives. If this is false, only sure matches will be replaced. Default: false
        replacementtext: The text to replace all matches with. If not set, the string "<address replaced>" will be used
        
    """
    url = f"https://esseguin-remove-contact-info.p.rapidapi.com/api.php"
    querystring = {'message': message, }
    if greedy:
        querystring['greedy'] = greedy
    if replacementtext:
        querystring['replacementText'] = replacementtext
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esseguin-remove-contact-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def remove_phone_numbers(message: str, greedy: bool=None, replacementtext: str='<phone number replaced>', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Remove all phone numbers from a given string of text. Matches formats such as: 555.666.7777, (555) 666-7777, 555 666 7777."
    message: This is a string which will be stripped of phone numbers
        greedy: Setting greedy to true will cause the replacement to be more likely to match false positives. If this is false, only sure matches will be replaced. Default: false
        replacementtext: The text to replace all matches with. If not set, the string "<phone number replaced>" will be used
        
    """
    url = f"https://esseguin-remove-contact-info.p.rapidapi.com/api.php"
    querystring = {'message': message, }
    if greedy:
        querystring['greedy'] = greedy
    if replacementtext:
        querystring['replacementText'] = replacementtext
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esseguin-remove-contact-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def remove_emails(message: str, greedy: bool=None, replacementtext: str='<email replaced>', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Remove all emails from a given string of text. This will match things such as: name@domain.com, name at domain dot com, name@domain dot com"
    message: This is a string which will be stripped of emails
        greedy: Setting greedy to true will cause the replacement to be more likely to match false positives. If this is false, only sure matches will be replaced. Default: false
        replacementtext: The text to replace all matches with. If not set, the string "<email replaced>" will be used
        
    """
    url = f"https://esseguin-remove-contact-info.p.rapidapi.com/api.php"
    querystring = {'message': message, }
    if greedy:
        querystring['greedy'] = greedy
    if replacementtext:
        querystring['replacementText'] = replacementtext
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esseguin-remove-contact-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def remove_all(message: str, greedy: bool=None, replacementtext: str='<contact information replaced>', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Remove all phone numbers, addresses, and emails from a given string of text"
    message: This is a string which will be stripped of all contact information
        greedy: Setting greedy to true will cause the replacement to be more likely to match false positives. If this is false, only sure matches will be replaced. Default: false
        replacementtext: The text to replace all matches with. If not set, the string "<contact information replaced>" will be used
        
    """
    url = f"https://esseguin-remove-contact-info.p.rapidapi.com/api.php"
    querystring = {'message': message, }
    if greedy:
        querystring['greedy'] = greedy
    if replacementtext:
        querystring['replacementText'] = replacementtext
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esseguin-remove-contact-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def remove_all_multiple(messages: str, greedy: bool=None, replacementtext: str='<contact information replaced>', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Remove all contact information from multiple messages. This endpoint lets you batch conversions and is faster than multiple individual requests."
    messages: Multiple URL encoded strings, separated with commas
        greedy: Setting greedy to true will cause the replacement to be more likely to match false positives. If this is false, only sure matches will be replaced. Default: false
        replacementtext: The text to replace all matches with. If not set, the string "<contact information replaced>" will be used
        
    """
    url = f"https://esseguin-remove-contact-info.p.rapidapi.com/api.php"
    querystring = {'messages': messages, }
    if greedy:
        querystring['greedy'] = greedy
    if replacementtext:
        querystring['replacementText'] = replacementtext
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esseguin-remove-contact-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

