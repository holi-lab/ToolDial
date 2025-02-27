import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def countcompanies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# Description GET Number Of Companies
		
		This url is to get the number of Companies created is type `GET`, the url is `BASE_URL/companies/count` and returns the number of Companies."
    
    """
    url = f"https://acopaer.p.rapidapi.com/companies/count"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "acopaer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def formby_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# Description GET Form By id
		
		This url is to get a Form by id is type `GET`, the url is `BASE_URL/forms/id` and returns only an object.
		
		The param is `id` must type there the id of Form to get details about this Form."
    
    """
    url = f"https://acopaer.p.rapidapi.com/forms/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "acopaer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countforms(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# Description GET Number Of Forms
		
		This url is to get the number of Forms created is type `GET`, the url is `BASE_URL/forms/count` and returns the number of Forms."
    
    """
    url = f"https://acopaer.p.rapidapi.com/forms/count"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "acopaer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def allcompanies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# Description GET All Companies
		
		This url is to get all Companies created is type `GET`, the url is `BASE_URL/companies` and returns an array of objects or only an object depends on how many are there.
		
		Click [here](https://strapi.io/documentation/v3.x/content-api/parameters.html#available-operators) for more info about params."
    
    """
    url = f"https://acopaer.p.rapidapi.com/companies/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "acopaer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def companyby_tax_number(tax_number: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# Description GET Company By tax_number
		
		This url is to get a Company by tax_number is type `GET`, the url is `BASE_URL/companies/tax_number` and returns only an object.
		
		The param is `tax_number` must type there the tax_number of Company to get details about this Company."
    
    """
    url = f"https://acopaer.p.rapidapi.com/companies/{tax_number}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "acopaer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def scenarioby_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# Description GET Scenario By id
		
		This url provides a Scenario by id, method type `GET`.
		
		`BASE_URL/scenarios/id`
		
		returns only an object.
		
		### Url parms:
		* `id`: scenario id"
    
    """
    url = f"https://acopaer.p.rapidapi.com/scenarios/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "acopaer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def allscenarios(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# Description GET All Scenarios
		
		This url provides all Scenarios created, method type `GET`.
		
		 `BASE_URL/scenarios`
		 
		 returns an array of objects or only an object depends on how many are there.
		
		
		Click [here](https://strapi.io/documentation/v3.x/content-api/parameters.html#available-operators) for more info about params."
    
    """
    url = f"https://acopaer.p.rapidapi.com/scenarios"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "acopaer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countanswersforms(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# Description GET Number Of Answers Forms
		
		This url is to get the number of Answers Forms created is type `GET`, the url is `BASE_URL/answers-forms/count` and returns the number of Answers Forms."
    
    """
    url = f"https://acopaer.p.rapidapi.com/answers-forms/count"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "acopaer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def allforms(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# Description GET All Forms
		
		This url is to get all Forms created is type `GET`, the url is `BASE_URL/forms` and returns an array of objects or only an object depends on how many are there.
		
		Click [here](https://strapi.io/documentation/v3.x/content-api/parameters.html#available-operators) for more info about params."
    
    """
    url = f"https://acopaer.p.rapidapi.com/forms/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "acopaer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def formactionby_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# Description GET Form Action By id
		
		This url is to get a Form Action by id is type `GET`, the url is `BASE_URL/form-actions/id` and returns only an object.
		
		The param is `id` must type there the id of Form Action to get details about this Form Action."
    
    """
    url = f"https://acopaer.p.rapidapi.com/form-actions/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "acopaer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reviewerby_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# Description GET Reviewer By id
		
		This url is to get a Reviewer by id is type `GET`, the url is `BASE_URL/reviewers/id` and returns only an object.
		
		The param is `id` must type there the id of Reviewer to get details about this Reviewer."
    
    """
    url = f"https://acopaer.p.rapidapi.com/reviewers/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "acopaer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def allreviewers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# Description GET All Reviewers
		
		This url is to get all Reviewers created is type `GET`, the url is `BASE_URL/reviewers` and returns an array of objects or only an object depends on how many are there.
		
		Click [here](https://strapi.io/documentation/v3.x/content-api/parameters.html#available-operators) for more info about params."
    
    """
    url = f"https://acopaer.p.rapidapi.com/reviewers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "acopaer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def allformactions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# Description GET All Form Actions
		
		This url is to get all Form Actions created is type `GET`, the url is `BASE_URL/form-actions` and returns an array of objects or only an object depends on how many are there.
		
		Click [here](https://strapi.io/documentation/v3.x/content-api/parameters.html#available-operators) for more info about params."
    
    """
    url = f"https://acopaer.p.rapidapi.com/form-actions/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "acopaer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def answerformby_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# Description GET Answer Form By id
		
		This url is to get a Answer Form by id is type `GET`, the url is `BASE_URL/answers-forms/id` and returns only an object.
		
		The param is `id` must type there the id of Answer Form to get details about this Answer Form."
    
    """
    url = f"https://acopaer.p.rapidapi.com/answers-forms/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "acopaer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def allanswersforms(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# Description GET All Answers Forms
		
		This url is to get all Answers Forms created is type `GET`, the url is `BASE_URL/answers-forms` and returns an array of objects or only an object depends on how many are there.
		
		Click [here](https://strapi.io/documentation/v3.x/content-api/parameters.html#available-operators) for more info about params."
    
    """
    url = f"https://acopaer.p.rapidapi.com/answers-forms"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "acopaer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countscenarios(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# Description GET Number Of Scenarios
		
		This url provides the Scenarios quantity created, method type `GET`.
		
		`BASE_URL/scenarios/count`
		
		returns the Scenarios quantity."
    
    """
    url = f"https://acopaer.p.rapidapi.com/scenarios/count"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "acopaer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countformactions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# Description GET Number Of Form Actions
		
		This url is to get the number of Form Actions created is type `GET`, the url is `BASE_URL/form-actions/count` and returns the number of Form Actions."
    
    """
    url = f"https://acopaer.p.rapidapi.com/form-actions/count"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "acopaer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countreviewers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# Description GET Number Of Reviewers
		
		This url is to get the number of Reviewers created is type `GET`, the url is `BASE_URL/reviewers/count` and returns the number of Reviewers."
    
    """
    url = f"https://acopaer.p.rapidapi.com/reviewers/count"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "acopaer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

