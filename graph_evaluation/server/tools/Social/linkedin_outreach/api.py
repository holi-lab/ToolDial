import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def linkedin_connections(urls: str, message: str, cookie: str, email: str, key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API does the sending connections job easy for you , just feed the URLs of the people you want to connect with and leave the job to us."
    urls: URLs of the people you wish to send requests, Please insert a insert a string of 10 URLs separated by comma. Example :    URL1, URL2, URL3, ....... , URL10  
        message: The message you will leave while sending request.  You can customize the message by using variables. For example, \"Hello [target_name],  would love to connect with you.\" is the message you want to send, here target_name is a variable, written in squared brackets '[]'. Assuming the target's name is John Wick then the resulting string will be, \"Hello John Wick, would love to connect with you.\"
Note: Please take care of the spelling while adding the variable(s).
        cookie: The value of cookie named \"li_at\".
Note: The above value is just a sample. To get the value actual value of the cookie , please follow the document link given.
        email: The email where you will receive the results of connection requests' status.
        
    """
    url = f"https://linkedin-outreach.p.rapidapi.com/send-requests/"
    querystring = {'urls': urls, 'message': message, 'cookie': cookie, 'email': email, 'key': key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "linkedin-outreach.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def linkedin_contacts(name: str, key: str, email: str, depth: int, designation: str='CEO', location: str='USA', company: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API fetches the Linkedin URL and Name of the search you have made."
    name: The domain or the company domain/name will work. Automobile, Software, HealthCare, Edutech are some examples.
        key: Use this key for testing.
        email: The email where you will receive the response at.
        depth: Depth tells the API to fetch info from number of Pages. If depth is 3 , the information will be collected from first 3 pages.
        designation: If you wish to get details of some particular designations only, please fill only one designation at a time. CEO , CTO , CFO , Director and so on.
        location: Location of the company or location of the Domain.
        company: For that particular domain , please mention a Company name if you wish to get details of the employees of that company.
        
    """
    url = f"https://linkedin-outreach.p.rapidapi.com/get-info/"
    querystring = {'name': name, 'key': key, 'email': email, 'depth': depth, }
    if designation:
        querystring['designation'] = designation
    if location:
        querystring['location'] = location
    if company:
        querystring['company'] = company
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "linkedin-outreach.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

