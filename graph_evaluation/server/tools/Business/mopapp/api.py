import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_sales(account: str, key: str, version: str, apporinappsales: int=0, application: int=129, country: str='US', currency: str='USD', datefrom: str='2011-01-01', dateto: str='2010-06-10', page: int=2, platform: int=2, store: int=4, storeaccount: int=4, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Extract the sales of your applications in raw format"
    account: your account name, chosen when you signed-up
        key: your secret API key, to be found under https://www.mopapp.com/my/api
        version: the version of the API function. Currently it's just 1.0
        apporinappsales: to include app and/or in-app sales. Possible values:
0 to include both app and in-app sales
1 to include only app sales
2 to include only in-app purchases
        application: the application ID, as assigned by Mopapp once you register the application. To find out this ID, go to My Account => Applications and copy the value of the "Mopapp ID" column for the desired application.
        country: the two-letter code representing the customer's country. For example US for USA, IT for Italy, DE for Germany etc.
        currency: the three-letter code for the currency used in the sale. Possible values are USD, EUR, GBP, CAD, JPY, AUD and INR. Note that the currency is totally independent from the country: so for example an American customer might purchase in USD, but a German customer might purchase in USD too, not necessarily in EUR (it usually depends on the store settings and the customer preferences).
        datefrom: the start date of the time range to include in the result set, in one of the following formats: yyyy-MM-dd HH:mm:ss or yyyy-MM-dd
        dateto: the end date of the time range to include in the result set, in one of the following formats: yyyy-MM-dd HH:mm:ss or yyyy-MM-dd
        page: since the API returns a maximus of 500 records per call, the result set is paginated, and this parameter is the number (1-based) of the page to retrieve. It's 1 by default. The totalPages parameter in the output will tell you the total number of pages you can retrieve with the specified filters.
        platform: the type of the platform to filter for. Possible values:
0 for Custom
1 for BlackBerry
2 for iPhone
3 for Android
4 for Symbian
5 for Pre
6 for Windows Phone
7 for iPad
8 for iPhone + iPad
        store: the type of the store to filter for. Possible values:
0 for Custom
1 for Handango
2 for MobiHand
3 for RIM App World
4 for Apple iTunes
5 for Android Market (paid apps)
6 for GetJar
7 for Android Market (free apps)
8 for WP7 Marketplace
9 for Amazon Appstore
        storeaccount: the store account ID, as assigned by Mopapp when you register it. To find out this ID, go to My Account => Store Accounts and copy the value of the "Mopapp ID" column for the desired account.
        
    """
    url = f"https://mopapp-mopapp.p.rapidapi.com/sales/list.json"
    querystring = {'account': account, 'key': key, 'version': version, }
    if apporinappsales:
        querystring['appOrInAppSales'] = apporinappsales
    if application:
        querystring['application'] = application
    if country:
        querystring['country'] = country
    if currency:
        querystring['currency'] = currency
    if datefrom:
        querystring['dateFrom'] = datefrom
    if dateto:
        querystring['dateTo'] = dateto
    if page:
        querystring['page'] = page
    if platform:
        querystring['platform'] = platform
    if store:
        querystring['store'] = store
    if storeaccount:
        querystring['storeAccount'] = storeaccount
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mopapp-mopapp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

