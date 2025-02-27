import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def list_of_all_categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all Categories:
		
		```
		    {
		        "id": 3,
		        "name": "ATV"
		    },
		    {
		        "id": 4,
		        "name": "Allround"
		    },
		    {
		        "id": 12,
		        "name": "Classic"
		    },
		    {
		        "id": 8,
		        "name": "Cross / motocross"
		    },
		    {
		        "id": 7,
		        "name": "Custom / cruiser"
		    },
		    {
		        "id": 14,
		        "name": "Enduro / offroad"
		    },
		    {
		        "id": 11,
		        "name": "Minibike, cross"
		    },
		    {
		        "id": 10,
		        "name": "Minibike, sport"
		    },
		    {
		        "id": 6,
		        "name": "Naked bike"
		    },
		    {
		        "id": 1,
		        "name": "Prototype / concept model"
		    },
		    {
		        "id": 5,
		        "name": "Scooter"
		    },
		    {
		        "id": 18,
		        "name": "Speedway"
		    },
		    {
		        "id": 2,
		        "name": "Sport"
		    },
		    {
		        "id": 17,
		        "name": "Sport touring"
		    },
		    {
		        "id": 9,
		        "name": "Super motard"
		    },
		    {
		        "id": 13,
		        "name": "Touring"
		    },
		    {
		        "id": 15,
		        "name": "Trial"
		    },
		    {
		        "id": 16,
		        "name": "Unspecified category"
		    }
		```"
    
    """
    url = f"https://motorcycle-specs-database.p.rapidapi.com/category"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motorcycle-specs-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def models_by_make_id_year(year: int, make: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "api/v1/model/make-id/{make}/{year}
		Get model by make id and year
		sample:
		
		```
		    {
		        "modelId": 1515,
		        "modelName": "C 400 GT",
		        "yearName": 2019,
		        "articleId": 2993
		    },
		    {
		        "modelId": 1516,
		        "modelName": "C 400 X",
		        "yearName": 2019,
		        "articleId": 2995
		    },
		    {
		        "modelId": 1518,
		        "modelName": "C 650 GT",
		        "yearName": 2019,
		        "articleId": 3002
		    },
		    {
		        "modelId": 1519,
		        "modelName": "C 650 GT Highline",
		        "yearName": 2019,
		        "articleId": 3010
		    },
		    {
		        "modelId": 1521,
		        "modelName": "C 650 Sport",
		        "yearName": 2019,
		        "articleId": 3014
		    },
		```"
    
    """
    url = f"https://motorcycle-specs-database.p.rapidapi.com/model/make-id/{make}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motorcycle-specs-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def image_by_id_as_link(article: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "/api/v1/article/{article}/image/link
		[ArticleGetImageByType]
		Get article image link
		sample:
		
		```
		{
		    "imageName": "BMW HP4 2015.jpg",
		    "link": "http://127.0.0.1:8000/files/BMW/2015/HP4/BMW_2015_HP4.jpg"
		}
		```"
    
    """
    url = f"https://motorcycle-specs-database.p.rapidapi.com/article/{article}/image/link"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motorcycle-specs-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def models_by_make_id_year_category(year: int, category: str, make: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get model by make ID, year and category:
		
		ex: /api/v1/model/make-id/55/2010/Sport"
    
    """
    url = f"https://motorcycle-specs-database.p.rapidapi.com/model/make-id/{make}/{year}/{category}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motorcycle-specs-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def models_by_make(make: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all models by make name
		ex: /model/make-name/BMW
		sample:
		
		```
		    {
		        "id": 1514,
		        "name": "450 Sports Enduro"
		    },
		    {
		        "id": 1515,
		        "name": "C 400 GT"
		    },
		    {
		        "id": 1516,
		        "name": "C 400 X"
		    },
		    {
		        "id": 1517,
		        "name": "C 600 Sport"
		    },
		    {
		        "id": 1518,
		        "name": "C 650 GT"
		    },
		```"
    
    """
    url = f"https://motorcycle-specs-database.p.rapidapi.com/model/make-name/{make}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motorcycle-specs-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def models_by_make_id_and_category(make: int, category: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all models  by make ID and category
		ex: /api/v1/model/make-id/100/category/Sport
		
		sample:
		
		```
		    {
		        "modelId": 2713,
		        "modelName": "Altino 125 ES",
		        "yearName": 2004,
		        "categoryName": "Sport",
		        "priceName": null,
		        "articleId": 5559
		    },
		    {
		        "modelId": 2730,
		        "modelName": "Daystar 125 FI",
		        "yearName": 2011,
		        "categoryName": "Sport",
		        "priceName": " Euro 2990.  Prices depend on country, taxes, accessories, etc.",
		        "articleId": 5590
		    },
		    {
		        "modelId": 2745,
		        "modelName": "RoadSport",
		        "yearName": 2015,
		        "categoryName": "Sport",
		        "priceName": null,
		        "articleId": 5610
		    }
		```"
    
    """
    url = f"https://motorcycle-specs-database.p.rapidapi.com/model/make-id/{make}/category/{category}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motorcycle-specs-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_all_makes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of all Makes
		[MakeAll]
		Get a list of all makes
		sample:
		
		```
		   {
		        "id": "55",
		        "name": "BMW"
		    },
		    {
		        "id": "60",
		        "name": "BPG"
		    },
		    {
		        "id": "66",
		        "name": "BSA"
		    }
		```"
    
    """
    url = f"https://motorcycle-specs-database.p.rapidapi.com/make"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motorcycle-specs-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def specifications_by_make_model(make: str, model: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Specifications by {Make} / {Model}
		[MakeModelGetCompleteSpecification]"
    
    """
    url = f"https://motorcycle-specs-database.p.rapidapi.com/make/{make}/model/{model}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motorcycle-specs-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def specifications_by_year_make_model(make: str, model: str, year: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GET /article/{year}/{make}/{model}
		[ArticleGetByYearMakeModel]"
    
    """
    url = f"https://motorcycle-specs-database.p.rapidapi.com/article/{year}/{make}/{model}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motorcycle-specs-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def production_years_grouped_by_model_id(modelid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get makeName, modelName, years, by model Id"
    
    """
    url = f"https://motorcycle-specs-database.p.rapidapi.com/model/{modelid}/grouped-models-years"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motorcycle-specs-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def production_years_by_model_id(modelid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get makeName, modelName, years, by model Id"
    
    """
    url = f"https://motorcycle-specs-database.p.rapidapi.com/model/{modelid}/years"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motorcycle-specs-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def specifications_by_group(specs: str, article: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "ArticleGetSpecificationGroup
		{specs} => engineAndTransmission
		{specs} => chassisSuspensionBrakesAndWheels
		{specs} => physicalMeasuresAndCapacities
		{specs} => otherSpecifications"
    
    """
    url = f"https://motorcycle-specs-database.p.rapidapi.com/article/{article}/specs/{specs}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motorcycle-specs-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def models_by_make_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all model list by make ID
		/model/make-id/{id}
		
		sample:
		
		```
		    {
		        "id": 1514,
		        "name": "450 Sports Enduro"
		    },
		    {
		        "id": 1515,
		        "name": "C 400 GT"
		    },
		    {
		        "id": 1516,
		        "name": "C 400 X"
		    },
		    {
		        "id": 1517,
		        "name": "C 600 Sport"
		    },
		    {
		        "id": 1518,
		        "name": "C 650 GT"
		    },
		    {
		        "id": 1519,
		        "name": "C 650 GT Highline"
		    },
		    {
		        "id": 1520,
		        "name": "C 650 GT Highline LS"
		    },
		```"
    
    """
    url = f"https://motorcycle-specs-database.p.rapidapi.com/model/make-id/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motorcycle-specs-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def image_by_id_as_media_content(article: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get article image as media as content.
		[ArticleGetImageByType]
		Useful for mobile apps."
    
    """
    url = f"https://motorcycle-specs-database.p.rapidapi.com/article/{article}/image/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motorcycle-specs-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def specifications_by_id(article: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all specifications available by ID
		[ArticleGetCompleteSpecification]"
    
    """
    url = f"https://motorcycle-specs-database.p.rapidapi.com/article/{article}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motorcycle-specs-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

