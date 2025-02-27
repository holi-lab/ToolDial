import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getdealersbyid(dealerid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Premium. Dealership information using the internal ID. Returns name, address, state, zipCode, and ID for a single dealer in the same format as the /getDealers endpoint. Dealer IDs are generally retrieved via the /getDealers endpoint by zipcode"
    
    """
    url = f"https://cis-automotive.p.rapidapi.com/getDealersByID"
    querystring = {'dealerID': dealerid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cis-automotive.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdealers(zipcode: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Dealership information in a given zip code using the first 4 digits. Returns name, address, state, zipCode.     For example a call with the zip code 92701 would return dealers in the range [92700, 92709]"
    
    """
    url = f"https://cis-automotive.p.rapidapi.com/getDealers"
    querystring = {'zipCode': zipcode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cis-automotive.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdealersbyregion(regionname: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Premium. Dealership information in a given region. Returns name, address, state, zipCode, and IDs. Results are paginated with up to 30 results per page."
    
    """
    url = f"https://cis-automotive.p.rapidapi.com/getDealersByRegion"
    querystring = {'regionName': regionname, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cis-automotive.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def valuation(vin: str, radius: int=None, dealerid: int=None, newcars: bool=None, sameyear: bool=None, daysback: int=45, longitude: int=None, latitude: int=None, regionname: str='REGION_STATE_NC', startdate: str=None, enddate: str=None, mileagelow: int=None, extendedsearch: bool=None, mileagehigh: int=None, zipcode: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Premium. Provides the average, stdDev, and count, of the sale price and mileage of similar new or used vehicles based off the provided VIN and matching the provided other search criteria. This endpoint can be easily used to determine market values in arbitrary geographic locations (like a city) for specific vehicles. See /listings2 endpoint for documentation on location, vehicle, and time search parameters. Date selection is restricted by your subscription tier, same as with the /listings2 endpoint. Optionally restricts report to vehicles of the same model year."
    
    """
    url = f"https://cis-automotive.p.rapidapi.com/valuation"
    querystring = {'vin': vin, }
    if radius:
        querystring['radius'] = radius
    if dealerid:
        querystring['dealerID'] = dealerid
    if newcars:
        querystring['newCars'] = newcars
    if sameyear:
        querystring['sameYear'] = sameyear
    if daysback:
        querystring['daysBack'] = daysback
    if longitude:
        querystring['longitude'] = longitude
    if latitude:
        querystring['latitude'] = latitude
    if regionname:
        querystring['regionName'] = regionname
    if startdate:
        querystring['startDate'] = startdate
    if enddate:
        querystring['endDate'] = enddate
    if mileagelow:
        querystring['mileageLow'] = mileagelow
    if extendedsearch:
        querystring['extendedSearch'] = extendedsearch
    if mileagehigh:
        querystring['mileageHigh'] = mileagehigh
    if zipcode:
        querystring['zipCode'] = zipcode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cis-automotive.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbrands(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get vehicle brand names. These names are used as arguments for other endpoints. The names are generally not case sensitive
		when used with other endpoints, but it is best practice to use the names returned by this endpoint without changes."
    
    """
    url = f"https://cis-automotive.p.rapidapi.com/getBrands"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cis-automotive.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbrands(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get vehicle brand names. These names are used as arguments for other endpoints. The names are generally not case sensitive
		when used with other endpoints, but it is best practice to use the names returned by this endpoint without changes."
    
    """
    url = f"https://cis-automotive.p.rapidapi.com/getBrands"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cis-automotive.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listings2(latitude: int=0, longitude: int=0, dealerid: int=0, enddate: str=None, mileagelow: int=0, newcars: bool=None, startdate: str=None, modelname: str='F-150', brandname: str='Ford', extendedsearch: bool=None, mileagehigh: int=0, page: int=1, daysback: int=45, zipcode: int=0, regionname: str='REGION_STATE_VA', modelyear: int=0, radius: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generic getter for listings supporting a wide array of selection criteria. This is the new primary listing endpoint and we will phase out the older listing endpoints over time. The other listing endpoints return the same data, but are more restrictive in the available geographic and vehicle selection criteria and can be replicated by this endpoint.
		
		Dealer selection uses the most restrictive criteria supplied. From most restrictive to least: dealerID, gps, zipCode, region. You must provide some dealer selection criteria.
		
		It is important to note that the units in the longitude are in degrees east, not degrees west. For example the coordinates 45.53N, 100.41W correspond to Mobridge, SC but they will be interpreted as 45.53N, 100.41E which corresponds to a point in the Gobi Desert near Jinst, Mongolia. You can fix this by converting the longitiude yourself, or by supplying a negative value (-100.41). For this example both (X, -100.41) and (X, 259.59) would be the same point. Units on the radius are miles and a smaller radius will result in a faster response time. Maximum search radius depends on your subscription plan. The radius value is only used for GPS searches and is ignored for searches using other location criteria.
		
		Listing selection logically ANDs all options given.
		
		Time interval selection will prefer explicit start and end dates. If only one of startDate/endDate is supplied, this endpoint will use it as an anchor and look forward or backwards by the daysBack value. If startDate is specified and endDate is not, then endDate will be set to startDate+daysBack. Conversely if endDate is specified, but startDate is not then startDate will be set at endDate-daysBack. If neither is supplied endpoint will set endDate to today and startDate to today-daysBack. This endpoint is restricted to listings from the last 90 days on RapidAPI.
		
		Maximum time interval is 45 days.
		
		Mileage selection uses the provided mileage values and returns vehicles with mileage in the range [mileageLow, mileageHigh]. If mileageLow == mileageHigh (for example both are 0 default) this endpoint will not filter based on mileage. Not all used vehicles have a mileage record available.
		
		ExtendedSearch modifies the slice of listings returned. If false (default) it only returns vehicles satisfying lastSeen >= startDate and lastSeen < endDate. If true it will return vehicles that were in dealer's inventory at any point between startDate and endDate including vehicles that were sold after endDate. Setting extendedSearch to true will result in a slower response time.
		
		For example: If both a region name and dealer ID are supplied the dealer ID will be used because it is the most restrictive.
		
		If a brandName of Ford and modelYear of 2019, modelName of F-150, and newCars of False is supplied this endpoint will return used 2019 model year Ford F-150s. If a contradictory listing selection is supplied (for example Ford + Camry) no listings will be returned because the request matched no listings.
		
		Results are paginated in chunks of up to 20 vehicles. Prices are in the dealer's local currency (generally USD)."
    
    """
    url = f"https://cis-automotive.p.rapidapi.com/listings2"
    querystring = {}
    if latitude:
        querystring['latitude'] = latitude
    if longitude:
        querystring['longitude'] = longitude
    if dealerid:
        querystring['dealerID'] = dealerid
    if enddate:
        querystring['endDate'] = enddate
    if mileagelow:
        querystring['mileageLow'] = mileagelow
    if newcars:
        querystring['newCars'] = newcars
    if startdate:
        querystring['startDate'] = startdate
    if modelname:
        querystring['modelName'] = modelname
    if brandname:
        querystring['brandName'] = brandname
    if extendedsearch:
        querystring['extendedSearch'] = extendedsearch
    if mileagehigh:
        querystring['mileageHigh'] = mileagehigh
    if page:
        querystring['page'] = page
    if daysback:
        querystring['daysBack'] = daysback
    if zipcode:
        querystring['zipCode'] = zipcode
    if regionname:
        querystring['regionName'] = regionname
    if modelyear:
        querystring['modelYear'] = modelyear
    if radius:
        querystring['radius'] = radius
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cis-automotive.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listings(newcars: bool, dealerid: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a dealer's listings over the last 45 days. Listing keys are: vin, askPrice, msrp, isNew, firstSeen, lastSeen, modelName, brandName. Results are paginated in chunks of up to 20 vehicles. Prices are in the dealer's local currency (generally USD)."
    
    """
    url = f"https://cis-automotive.p.rapidapi.com/listings"
    querystring = {'newCars': newcars, 'dealerID': dealerid, 'page': page, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cis-automotive.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listingsbyregion(regionname: str, modelname: str, newcars: bool=None, daysback: int=45, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a dealer's listings over up to the last 45 days by dealer ID. The ID can be found by calling the /getDealers endpoint. Listing keys are: vin, askPrice, msrp, isNew, firstSeen, lastSeen, modelName, brandName. Results are paginated in chunks of up to 20 vehicles. Prices are in the dealer's local currency (generally USD)."
    
    """
    url = f"https://cis-automotive.p.rapidapi.com/listingsByRegion"
    querystring = {'regionName': regionname, 'modelName': modelname, }
    if newcars:
        querystring['newCars'] = newcars
    if daysback:
        querystring['daysBack'] = daysback
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cis-automotive.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getregions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get region names. These names are used as arguments for other endpoints. The names are generally not case sensitive
		when used with other endpoints, but it is best practice to use the names returned by this endpoint without changes."
    
    """
    url = f"https://cis-automotive.p.rapidapi.com/getRegions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cis-automotive.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getinactivemodels(brandname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all model names including discontinued models. Because these models are no longer built, or have very poor market performance they are not incuded in the normal getModels endpoint. Many users itterate through the model names with our new vehicle sales endpoints and waste some of their quota making self contradictory requests. This endpoint was created to aleviate the use case where someone requests information on new vehicle sales for a model that has not been sold new for a long, long, time.  These names are used as arguments for other endpoints. The names are generally not case sensitive when used with other endpoints, but it is best practice to use the names returned by this endpoint without changes."
    
    """
    url = f"https://cis-automotive.p.rapidapi.com/getInactiveModels"
    querystring = {'brandName': brandname, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cis-automotive.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmodels(brandname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get brand model names for currently active models. This endpoint does not return model names that have been discontinued or have sold less than 10 vehicles in the last month and a half.  These names are used as arguments for other endpoints. The names are generally not case sensitive when used with other endpoints, but it is best practice to use the names returned by this endpoint without changes."
    
    """
    url = f"https://cis-automotive.p.rapidapi.com/getModels"
    querystring = {'brandName': brandname, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cis-automotive.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def salepricehistogram(modelname: str, brandname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Histogram of the sale price of vehicles over the last 45 days for a given model and region. Price buckets are grouped in units of $1000 The available brand, model, and region names can be retrieved from their respective endpoints."
    
    """
    url = f"https://cis-automotive.p.rapidapi.com/salePriceHistogram"
    querystring = {'modelName': modelname, 'brandName': brandname, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cis-automotive.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dayssupply(brandname: str, regionname: str='REGION_STATE_CA', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Average, median, standard deviation, population variance, and whole region average of the 
		days of supply left on dealer lots for a given brand and region. The average, median, stdDev, and pVar fields are calculated on
		a dealer by dealer basis while the whole region average treats the entire region like a single dealership. 
		The average field may differ from the whole region average, especially when dealers are out of 
		a given model. 
		
		The available brand and region names can be retrieved from their respective endpoints."
    
    """
    url = f"https://cis-automotive.p.rapidapi.com/daysSupply"
    querystring = {'brandName': brandname, }
    if regionname:
        querystring['regionName'] = regionname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cis-automotive.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def daystosell(brandname: str, regionname: str='REGION_STATE_CA', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Average, median, standard deviation, population variance, and whole region average of the 
		number of days a vehicle spends on dealer lots for a given brand and region. The average, median, stdDev, and pVar fields are calculated on
		a dealer by dealer basis while the whole region average treats the entire region like a single dealership. 
		The average field may differ from the whole region average.
		
		The available brand and region names can be retrieved from their respective endpoints."
    
    """
    url = f"https://cis-automotive.p.rapidapi.com/daysToSell"
    querystring = {'brandName': brandname, }
    if regionname:
        querystring['regionName'] = regionname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cis-automotive.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listprice(brandname: str, regionname: str='REGION_STATE_CA', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Average, median, standard deviation, and population variance of the ask price of new vehicles over the last 15 days for a given brand and region.
		
		The available brand and region names can be retrieved from their respective endpoints."
    
    """
    url = f"https://cis-automotive.p.rapidapi.com/listPrice"
    querystring = {'brandName': brandname, }
    if regionname:
        querystring['regionName'] = regionname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cis-automotive.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def saleprice(brandname: str, regionname: str='REGION_STATE_CA', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Average, median, standard deviation, and population variance of the sale price of new vehicles over the last 15 days for a given brand and region.
		
		The available brand and region names can be retrieved from their respective endpoints."
    
    """
    url = f"https://cis-automotive.p.rapidapi.com/salePrice"
    querystring = {'brandName': brandname, }
    if regionname:
        querystring['regionName'] = regionname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cis-automotive.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def topmodels(regionname: str='REGION_STATE_CA', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Sales ranking of different models by region over the last 45 days. The percentOfTopSales value is the percent of the top seller the model represents. For example: a value of 80% means that model sold 8 vehicles for every 10 of the top model sold.The other fields represent the model percent of X. The <strong>brandMarketShare</strong> field is that brand's market share of the regionover the report's time interval."
    
    """
    url = f"https://cis-automotive.p.rapidapi.com/topModels"
    querystring = {}
    if regionname:
        querystring['regionName'] = regionname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cis-automotive.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def regiondailysales(day: str, brandname: str, regionname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get regional sales by brand and day. Most recent data is typically only 2 days old for this endpoint.
		
		The Day field is in YYYY-MM-DD format. For example if you wanted sales data from April 5th of 2020 the day field would be '2020-04-05'
		
		Data availability depends on region and goes back up to 2016."
    
    """
    url = f"https://cis-automotive.p.rapidapi.com/regionDailySales"
    querystring = {'day': day, 'brandName': brandname, 'regionName': regionname, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cis-automotive.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def regionsales(brandname: str, month: str, regionname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Premium. Get regional sales by brand and month, broken down by day. Most recent data is typically only 2 days old for this endpoint.
		
		The month field is in YYYY-MM-DD format. For example if you wanted sales data from April of 2020 the month field would be '2020-04-01'
		
		Data availability depends on region and goes back up to 2016."
    
    """
    url = f"https://cis-automotive.p.rapidapi.com/regionSales"
    querystring = {'brandName': brandname, 'month': month, 'regionName': regionname, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cis-automotive.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def vehiclehistory(vin: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides a simple report detailing a vechicle's sales history at dealerships. Data includes the name of the dealership, dates it was for sale, price, new/used condition, mileage, dealership state, and dealership zip code. The data for this endpoint is generally refreshed weekly."
    
    """
    url = f"https://cis-automotive.p.rapidapi.com/vehicleHistory"
    querystring = {'vin': vin, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cis-automotive.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def vindecode(vin: str, passempty: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Decodes the provided North American vin and provides recall information if available. We require at least the first 12 out of 17 characters in the vin to attempt a decode. The vin is not case sensitive. If passEmpty (default False) is True we will also include the empty fields in the response json."
    
    """
    url = f"https://cis-automotive.p.rapidapi.com/vinDecode"
    querystring = {'vin': vin, }
    if passempty:
        querystring['passEmpty'] = passempty
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cis-automotive.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def similarsaleprice(vin: str, regionname: str='REGION_STATE_CA', daysback: int=45, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides the average, stdDev, and count, of the sale price and mileage of similar new and used vehicles in a given region based off the provided VIN. Optionally restricts report to vehicles of the same model year and goes back up to 120 days."
    
    """
    url = f"https://cis-automotive.p.rapidapi.com/similarSalePrice"
    querystring = {'vin': vin, }
    if regionname:
        querystring['regionName'] = regionname
    if daysback:
        querystring['daysBack'] = daysback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cis-automotive.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

