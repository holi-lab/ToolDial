import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getrecipe(recipeid: str, servings: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the details for a specific recipe."
    recipeid: The id of the recipe to get data for.
        servings: Number of servings. Use "0" to obtain default servings used when this recipe was created.
        
    """
    url = f"https://dp.p.rapidapi.com/"
    querystring = {'RecipeId': recipeid, 'Servings': servings, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcategories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets a list of all categories."
    
    """
    url = f"https://dp.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def setfavoriterecipe(method: str, deviceid: str, userid: int, recipeid: int, isfavorite: bool, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Sets a user favorite recipe."
    method: The method to call on the server.
        deviceid: The unique id of the client calling the method.
        userid: The id of the user.
        recipeid: The id of the recipe to flag for favorite.
        isfavorite: Should the recipe be flagged as a favorite, 0 for false, 1 for true.
        
    """
    url = f"https://dp.p.rapidapi.com/"
    querystring = {'method': method, 'DeviceId': deviceid, 'UserId': userid, 'RecipeId': recipeid, 'IsFavorite': isfavorite, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmealboxcategories(targetstoreid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets all categories for meal boxes."
    targetstoreid: Identifies the categories to use for this request. If no recipes attached to a category, category will not be included in the result returned.
        
    """
    url = f"https://dp.p.rapidapi.com/"
    querystring = {'targetstoreid': targetstoreid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getfavoriterecipes(method: str, userid: int, deviceid: str, pageindex: int=0, pagesize: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the recipes the user has flagged as favorites."
    method: The name of the method to call.
        userid: The id of the user requesting the data.
        deviceid: The unique id of the device requesting the data.
        pageindex: The page index to get data for.
        pagesize: The number of recipes to return pr. page.
        
    """
    url = f"https://dp.p.rapidapi.com/"
    querystring = {'method': method, 'UserId': userid, 'deviceid': deviceid, }
    if pageindex:
        querystring['PageIndex'] = pageindex
    if pagesize:
        querystring['PageSize'] = pagesize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcollectedrecipes(rnd: str, weektoken: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the recipes the user has added to his meal plan."
    rnd: A random string to ensure you do not get a cached version from somewhere.
        weektoken: Can be used to automatic add recipes from some custom logic to this users meal plan before retrieving the collected recipes. Leave it blank.
        
    """
    url = f"https://dp.p.rapidapi.com/"
    querystring = {'rnd': rnd, }
    if weektoken:
        querystring['WeekToken'] = weektoken
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def setmealdate(collectid: int, mealdate: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Set the date and time for a recipe on the meal plan."
    collectid: The id of the record to make the change to.
        mealdate: the date and time in the format yyyy-MM-ddTHH:mm
        
    """
    url = f"https://dp.p.rapidapi.com/"
    querystring = {'CollectId': collectid, 'mealdate': mealdate, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlistsofvalues(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets list of values to be used with drop downs."
    
    """
    url = f"https://dp.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getrecipesbyfreetext(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets a list of recipes. Max number of recipes returned is limited to 50."
    q: The text to search for in the recipes.
        
    """
    url = f"https://dp.p.rapidapi.com/"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def addrecipetomealplan(servings: int, mealdate: str, recipeid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Adds a recipe to the users meal plan."
    servings: The amount of servings to add.
        mealdate: Use "0" to add to next free slot, or use format 2014-11-05T13:15:30.
        recipeid: The id of the recipe to add to the meal plan.
        
    """
    url = f"https://dp.p.rapidapi.com/"
    querystring = {'Servings': servings, 'MealDate': mealdate, 'recipeid': recipeid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettabrecipes(pagesize: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get 3 sets of recipe data (popular, new and recommended)"
    pagesize: The number of recipes to return in each dataset.
        
    """
    url = f"https://dp.p.rapidapi.com/"
    querystring = {'PageSize': pagesize, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def removecollectedrecipes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Removes all recipes form the meal plan."
    
    """
    url = f"https://dp.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdefaultsuppliermappingbyrecipe(recipeid: str, servings: int, departmentid: int, supplierid: int, supplierpriceconfiguration: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets a list of products that if purchased matches the ingredients attached to the recipe."
    recipeid: The id of the recipe to get the mapping for.
        servings: The amount of servings to calculate product mappings for. If 0, we use recipe default.
        departmentid: If the supplier has multiple departments for delivering groceries, this will identify the department.
        supplierid: The supplier to get the mapping for. This is only an option if the system you build should be connected to an online supermarket.
        supplierpriceconfiguration: The products to display for this recipe if multiple products have been attached to each ingredient. Possible values at the moment are: DefaultSelection = 0, CheapestPossible = 1, Organic = 2.
        
    """
    url = f"https://dp.p.rapidapi.com/"
    querystring = {'RecipeId': recipeid, 'Servings': servings, 'DepartmentId': departmentid, 'supplierid': supplierid, 'supplierpriceconfiguration': supplierpriceconfiguration, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getrecipesincategory(mealcatid: int, pageindex: int, pagesize: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the recipes for a specific category. TODO: Add current page and total number of pages to response."
    mealcatid: The id of the category to get the recipes for.
        pageindex: The page to retrieve from the server.
        pagesize: The number of recipes per page.
        
    """
    url = f"https://dp.p.rapidapi.com/"
    querystring = {'MealCatId': mealcatid, 'PageIndex': pageindex, }
    if pagesize:
        querystring['PageSize'] = pagesize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def setconfirmitemsreceivedatsupplier(orderiddraft: int, draftkey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Supermarket must call this method just after meal plan was transferred to basket from foreign domain."
    orderiddraft: The draft id created when converting the meal plan to a basket.
        draftkey: A uniqueidentifier in the format DF452C8E-DE33-425A-8958-77E697384928. This identifier is generated when meal plan is converted to ingredients.
        
    """
    url = f"https://dp.p.rapidapi.com/"
    querystring = {'OrderIdDraft': orderiddraft, 'DraftKey': draftkey, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmealboxrecipes(mealboxid: int, log: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets a list of recipes for a specific meal box."
    mealboxid: The id of the meal box to get the recipes for.
        log: Should the request be logged and data used in analytics such as most viewed meal boxes. Default is 0. Possible values 0 or 1.
        
    """
    url = f"https://dp.p.rapidapi.com/"
    querystring = {'MealBoxId': mealboxid, 'log': log, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def removerecipefrommealplan(collectid: int, recipeid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Removes a recipe form the meal plan."
    collectid: The id representing the row to delete in the DB.
        recipeid: The id of the recipe to remove from the meal plan.
        
    """
    url = f"https://dp.p.rapidapi.com/"
    querystring = {'CollectId': collectid, 'RecipeId': recipeid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def setconfirmorderfromsupplier(orderiddraft: int, transactionid: str, supplierordertotalprice: int, suppliernewcustomer: bool, draftkey: str, supplierorderid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Confirms that a meal plan was converted to an order and payment accepted."
    orderiddraft: An id generated when the meal plan is converted to ingredients, but before receiving payment.
        transactionid: The transaction id for the payment, used for tracking back.
        supplierordertotalprice: The total amount of the basket, which was paid in the transaction.
        suppliernewcustomer: A value indicating if this was the first order the customer placed at the online supermarket.
        draftkey: A uniqueidentifier in the format DF452C8E-DE33-425A-8958-77E697384928. This identifier is generated when meal plan is converted to ingredients.
        supplierorderid: The order id this transaction was given at the online supermarket. Used for tracking and linking data.
        
    """
    url = f"https://dp.p.rapidapi.com/"
    querystring = {'OrderIdDraft': orderiddraft, 'TransactionId': transactionid, 'SupplierOrderTotalPrice': supplierordertotalprice, 'SupplierNewCustomer': suppliernewcustomer, 'draftkey': draftkey, 'supplierorderid': supplierorderid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getgrocerylistbyuserid(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the sum of grocery list for the current user. If UserId is 0, data stored on DeviceId is used in stead. If UserId > 0 we check if DeviceId is associated with UserId on backend."
    
    """
    url = f"https://dp.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def setrecipecollectservings(collectid: int, servings: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Sets the number of servings on the meal plan for a given recipe."
    collectid: The id in the DB to update data for.
        servings: The number of servings to set for the recipe on the meal plan.
        
    """
    url = f"https://dp.p.rapidapi.com/"
    querystring = {'CollectId': collectid, 'Servings': servings, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmealboxesbycategoryid(targetstoreid: str, mealboxcategoryid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the meal boxes attached to a specific category."
    targetstoreid: The id of the store to get the list of meal boxes for.
        mealboxcategoryid: The id of the category to get the meal boxes for.
        
    """
    url = f"https://dp.p.rapidapi.com/"
    querystring = {'TargetStoreId': targetstoreid, 'MealBoxCategoryId': mealboxcategoryid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlistofingredientsusedinrecipes(q: str='kartof', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets a complete list of ingredients used in recipes."
    q: A text used in a like filter on the ingredients to return. Leave empty to return all ingredients.
        
    """
    url = f"https://dp.p.rapidapi.com/"
    querystring = {}
    if q:
        querystring['q'] = q
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getweeklymealplan(weeklyid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the recipes for a specific weekly meal plan."
    weeklyid: The id of the meal plan to get the recipes for. Use 0 for current week.
        
    """
    url = f"https://dp.p.rapidapi.com/"
    querystring = {'WeeklyId': weeklyid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

