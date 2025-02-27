import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def logo(domain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Logo API is used to show company logos in the frontend. The API link is already included in results returned by the Similar companies API."
    domain: (Required) The company domain for which you want to retrieve a logo for
        
    """
    url = f"https://comparable-companies.p.rapidapi.com/logo"
    querystring = {'domain': domain, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "comparable-companies.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def similar(domain: str, limit: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The key feature of the comparable-companies API. This endpoint will return a list of companies that are similar to the one that has been provided. The endpoint will also return information about the queried domain through the variable "frontend_company_info".
		
		Occasionally, the algorithm finds new information that will take longer to crawl and process. In these cases, it will return results with the "further_crawling" variable set to true, indicating that we will have better results available in the next 10-15 sec. There is a possibility to establish a WebSocket connection through the "connectionID" parameter to be notified of these new results. This is not covered in this documentation, but our engineers will happily assist you with it.
		
		Querying the same domain multiple times will only cost you a single search credit."
    domain: (Required) The company domain for which you want to look up similar companies for (e.g. unilever.com). Providing a URL works as well, we will then parse the domain on our end.
        limit: (Optional) Max number of results to return. A lower limit leads to faster response times. Results beyond that limit are included in the response field \"next_companies\" which only shows their domain name and similarity percentage.
        
    """
    url = f"https://comparable-companies.p.rapidapi.com/similar"
    querystring = {'domain': domain, 'limit': limit, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "comparable-companies.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rating(comp_company_id: str, rating: int, main_company_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enables the user to rate the output returned by the /similar API. It is advisable to refresh results after ratings have been submitted as the algorithm fine-tunes its output based on them. Re-searching a company is always free of charge.
		
		*Example*: If a user searches similar companies for "unilever.com", and the first result is "nestle.com", then the user is able to rate how similar those companies are on a scale from 0-10 through this API. If the user submits a rating of 0, that comparable company ("nestle.com") will not show up when searching the same company ("unilever.com") in the future. If the user submits a rating higher than 5, then the algorithm will fine-tune its output to include more companies similar to that one ("nestle.com").
		
		*Including user ratings in the output is completely optional*."
    comp_company_id: (Required) domain_name of the comparable company which is being rated
        rating: (Required) The rating given by the user ranging from 0 to 10. Ratings above 5 will fine-tune the algorithm to find more companies similar to that one.
        main_company_id: (Required) The company domain which was queried
        
    """
    url = f"https://comparable-companies.p.rapidapi.com/rating"
    querystring = {'comp_company_id': comp_company_id, 'rating': rating, 'main_company_id': main_company_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "comparable-companies.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def contacts(domain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of contact persons (incl. email addresses) at a given company.
		
		Querying the same domain multiple times will only cost a single search credit.
		
		Querying a domain that you have already queried through the /similar endpoint will not cost any credit."
    domain: (Required) The company domain for which you want to look up contacts for (e.g. unilever.com). Providing a URL works as well, we will then parse the domain on our end.
        
    """
    url = f"https://comparable-companies.p.rapidapi.com/contacts"
    querystring = {'domain': domain, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "comparable-companies.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

