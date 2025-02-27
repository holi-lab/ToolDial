import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getfilesunverified(userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of files  owned by the  user with the provided user ID where status is unverified. The API support pagination by specifying the page number using page query parameter and number of file using  pagination query parameter. If the pagination parameters are not defined the API will return the first eight files.
		
		The API will return an a error for the following cases: 
		- User ID not provided
		-  The user with the provided user ID is not found in the database. 
		-  The file with the provided image ID and owned by the provided user ID is not found in the database. 
		
		In case of error the API will return 400 status code with a detailed message.
		In case of success the API will return 200 status code and the list of files owned by the user with the provided user ID  in data field."
    
    """
    url = f"https://chaincerts1.p.rapidapi.com/api/users/getFilesUnverified"
    querystring = {'userId': userid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chaincerts1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmycertnotverified(userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of files owned by the user with the provided user ID not minted to the blockchain . The API support pagination by specifying the page number using page query parameter and number of file using pagination query parameter. If the pagination parameters are not defined the API will return the first eight files.
		
		The API will return an a error for the following cases:
		
		- user ID is not provided.
		- The user with the provided user ID is not found in the database.
		
		In case of error the API will return 400 status code with a detailed message.
		In case of success the API will return 200 status code, the list of files owned by the user with the provided user ID in data field, and confirmation message in message field."
    
    """
    url = f"https://chaincerts1.p.rapidapi.com/api/users/getMyCertNotVerified"
    querystring = {'userId': userid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chaincerts1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def filterforsalecertificates40(type: str, username: str=None, userid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of files for sale  where the owner id or owner username equal the provided owner ID/ username and the type equal the provided type. The API support pagination by specifying the page number using page query parameter and number of file using  pagination query parameter. If the pagination parameters are not defined the API will return the first 40 files.
		
		The API will return an a error for the following cases: 
		-  User id is not provided.
		-  No user with the provided user ID in the database.
		 
		
		In case of error the API will return 400 status code with a detailed message.
		In case of success the API will return 200 status code, the list of files for sales  in the data filed, and the total number of files for sale in totalCert field."
    type: painting and the sculpture
        username: Either username or userId is required
        userid: Either username or userId is required
        
    """
    url = f"https://chaincerts1.p.rapidapi.com/api/users/filterForSaleCertificates40"
    querystring = {'type': type, }
    if username:
        querystring['username'] = username
    if userid:
        querystring['userId'] = userid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chaincerts1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listofitemsoldquantitygrterthn0(username: str=None, userid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of files for sale where the owner id or owner username equal the provided owner ID/ username and the quantity is grater than zero. 
		The API will return an a error for the following cases:
		
		User id is not provided.
		No user with the provided user ID in the database.
		In case of error the API will return 400 status code with a detailed message.
		In case of success the API will return 200 status code, the list of files for sales in the data filed, and the total number of files for sale in total field."
    username: Either username or userId is required
        userid: Either username or userId is required
        
    """
    url = f"https://chaincerts1.p.rapidapi.com/api/users/listofitemsoldquantitygrterthn0"
    querystring = {}
    if username:
        querystring['username'] = username
    if userid:
        querystring['userId'] = userid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chaincerts1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getfilestransferred(userid: str, page: str=None, pagination: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of files previously owned by the  user with the provided user ID. The API support pagination by specifying the page number using page query parameter and number of file using  pagination query parameter. If the pagination parameters are not defined the API will return the first eight files.
		
		The API will return an a error for the following cases: 
		- User ID not provided
		-  The user with the provided user ID is not found in the database. 
		-  The file with the provided image ID and owned by the provided user ID is not found in the database. 
		
		In case of error the API will return 400 status code with a detailed message.
		In case of success the API will return 200 status code and the list of files owned by the user with the provided user ID  in data field."
    
    """
    url = f"https://chaincerts1.p.rapidapi.com/api/users/getFilesTransferred"
    querystring = {'userId': userid, }
    if page:
        querystring['page'] = page
    if pagination:
        querystring['pagination'] = pagination
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chaincerts1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def filterforsalecertificates(type: str, username: str=None, userid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of files for sale  where the owner id or owner username equal the provided owner ID/ username and the type equal the provided type. The API support pagination by specifying the page number using page query parameter and number of file using  pagination query parameter. If the pagination parameters are not defined the API will return the first eight files.
		
		The API will return an a error for the following cases: 
		-  User id is not provided.
		-  No user with the provided user ID in the database.
		 
		
		In case of error the API will return 400 status code with a detailed message.
		In case of success the API will return 200 status code, the list of files for sales  in the data filed, and the total number of files for sale in totalCert field."
    type: example: painting , digital art, merchandise, etc.
        
    """
    url = f"https://chaincerts1.p.rapidapi.com/api/users/filterForSaleCertificates"
    querystring = {'type': type, }
    if username:
        querystring['username'] = username
    if userid:
        querystring['userId'] = userid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chaincerts1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getuserbyid(userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user information by id"
    
    """
    url = f"https://chaincerts1.p.rapidapi.com/api/users/getUserById"
    querystring = {'userId': userid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chaincerts1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbalance(userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get balance in Matic of the logged in user"
    
    """
    url = f"https://chaincerts1.p.rapidapi.com/api/users/getBalance"
    querystring = {'userId': userid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chaincerts1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sendverificationcode(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Send a verification code to the user via email.
		
		The API will return an a error for the following cases:
		
		The user with the specified user ID is not found in the database
		Error storing the generated code to database.
		If there is no error, the verification code will be send to the user via email.
		
		In case of error the API will return 400 status code with a detailed message.
		In case of success the API will return 200 status code with a confirmation message."
    
    """
    url = f"https://chaincerts1.p.rapidapi.com/api/users/send-verification-code"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chaincerts1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmycertificates(userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of files owned by the  user with the provided user ID. The API support pagination by specifying the page number using page query parameter and number of file using  pagination query parameter. If the pagination parameters are not defined the API will return the first eight files.
		
		The API will return an a error for the following cases: 
		- User ID not provided
		-  The user with the provided user ID is not found in the database. 
		-  The file with the provided image ID and owned by the provided user ID is not found in the database. 
		
		In case of error the API will return 400 status code with a detailed message.
		In case of success the API will return 200 status code and the list of files owned by the user with the provided user ID  in data field."
    
    """
    url = f"https://chaincerts1.p.rapidapi.com/api/users/getMyCertificates"
    querystring = {'userId': userid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chaincerts1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def servercheck(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://chaincerts1.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chaincerts1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getforsalecertificatesbyusername(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of files for sale  where the owner username equal the provided  username. The API support pagination by specifying the page number using page query parameter and number of file using  pagination query parameter. If the pagination parameters are not defined the API will return the first eight files.
		
		The API will return an a error for the following cases: 
		-  Username is not provided.
		-  No user with the provided username in the database.
		 
		
		In case of error the API will return 400 status code with a detailed message.
		In case of success the API will return 200 status code, the list of files for sales  in the data filed, and the total number of files for sale in totalCert field."
    
    """
    url = f"https://chaincerts1.p.rapidapi.com/api/users/getForSaleCertificatesByUsername"
    querystring = {'username': username, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chaincerts1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sortforsalecertificates40(order: str, userid: str=None, username: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of files for sale  where the owner username equal the provided  username, sorted by price either ascending or descending . The API support pagination by specifying the page number using page query parameter and number of file using  pagination query parameter. If the pagination parameters are not defined the API will return the first forty files.
		
		The API will return an a error for the following cases: 
		-  Username or user id are not provided.
		-  No user found with the provided username is the database.
		-  Order value is not ascending or descending.
		 
		
		In case of error the API will return 400 status code with a detailed message.
		In case of success the API will return 200 status code, the list of files for sales  in the data filed, and the total number of files for sale in totalCert field."
    order: ascending or descending
        
    """
    url = f"https://chaincerts1.p.rapidapi.com/api/users/sortForSaleCertificates40"
    querystring = {'order': order, }
    if userid:
        querystring['userId'] = userid
    if username:
        querystring['username'] = username
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chaincerts1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sortforsalecertificates(order: str, username: str=None, userid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of files for sale  where the owner username equal the provided  username, sorted by price either ascending or descending . The API support pagination by specifying the page number using page query parameter and number of file using  pagination query parameter. If the pagination parameters are not defined the API will return the first eight files.
		
		The API will return an a error for the following cases: 
		-  Username or user id are not provided.
		-  No user found with the provided username is the database.
		-  Order value is not ascending or descending.
		 
		
		In case of error the API will return 400 status code with a detailed message.
		In case of success the API will return 200 status code, the list of files for sales  in the data filed, and the total number of files for sale in totalCert field."
    order: accepted values: ascending or descending.
        
    """
    url = f"https://chaincerts1.p.rapidapi.com/api/users/sortForSaleCertificates"
    querystring = {'order': order, }
    if username:
        querystring['username'] = username
    if userid:
        querystring['userId'] = userid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chaincerts1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def certminted(userid: str, enddate: str=None, subject: str=None, startdate: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get certificates that are minted and not transferred"
    enddate: eg 2022-01-04
        startdate: eg 2021-12-28
        
    """
    url = f"https://chaincerts1.p.rapidapi.com/api/users/certMinted"
    querystring = {'userId': userid, }
    if enddate:
        querystring['enddate'] = enddate
    if subject:
        querystring['subject'] = subject
    if startdate:
        querystring['startdate'] = startdate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chaincerts1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listdiscountcode(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://chaincerts1.p.rapidapi.com/api/users/listDiscountCode"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chaincerts1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcertificatesbyusername40(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://chaincerts1.p.rapidapi.com/api/users/getForSaleCertificatesByUsername40"
    querystring = {'username': username, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chaincerts1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getuserprofile(username: str=None, userid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://chaincerts1.p.rapidapi.com/api/users/getUserProfile"
    querystring = {}
    if username:
        querystring['username'] = username
    if userid:
        querystring['userId'] = userid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chaincerts1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcertificatesforsale(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of files for sale. The API support pagination by specifying the page number using page query parameter and number of file using  pagination query parameter. If the pagination parameters are not defined the API will return the first eight files.
		
		In case of error the API will return 400 status code with a detailed message.
		In case of success the API will return 200 status code, the list of files for sales  in the data filed, and the total number of files for sale in totalCert field."
    
    """
    url = f"https://chaincerts1.p.rapidapi.com/api/users/getCertificatesForSale"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chaincerts1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchtraits(traitvalue: str, userid: str=None, username: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of files for sale  where the owner id or owner username equal the provided owner ID/ username and the trait match one of the provided traits.
		
		The API will return an a error for the following cases: 
		-  User id or username is not provided.
		-  No user with the provided user ID in the database.
		 
		
		In case of error the API will return 400 status code with a detailed message.
		In case of success the API will return 200 status code, the list of files for sales  in the data filed, and the total number of files for sale in total field."
    userid: Either username or userId is required
        username: Either username or userId is required
        
    """
    url = f"https://chaincerts1.p.rapidapi.com/api/users/matchTraits"
    querystring = {'traitValue': traitvalue, }
    if userid:
        querystring['userId'] = userid
    if username:
        querystring['username'] = username
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chaincerts1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listoftheitemsoldandquantity0(userid: str=None, username: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of sold out  where the owner id or owner username equal the provided owner ID/ username. 
		The API will return an a error for the following cases:
		
		User id is not provided.
		No user with the provided user ID in the database.
		In case of error the API will return 400 status code with a detailed message.
		In case of success the API will return 200 status code, the list of files for sales in the data filed, and the total number of files for sale in total field."
    userid: Either userId or username is required
        username: Either userId or username is required
        
    """
    url = f"https://chaincerts1.p.rapidapi.com/api/users/listoftheitemsoldandquantity0"
    querystring = {}
    if userid:
        querystring['userId'] = userid
    if username:
        querystring['username'] = username
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chaincerts1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

