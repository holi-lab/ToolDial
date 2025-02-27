import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_mortgages(repaymentperiod: int, propertyvalue: int, mortgageamount: int, page: int=1, filterbyinitialperiod: str='twoYears,threeYears,fiveYears,tenYears', filterbyratetype: str='fixed,variable,discount,tracker', filterbypaymenttype: str='repayment', paginationlimit: int=12, sorttype: str='initialRate', location: str='england', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search All mortgages 
		
		Params:"
    filterbyinitialperiod: Comma separated list. Possible Values: `twoYears`,`threeYears`,`fiveYears`,`tenYears`
        filterbyratetype: Comma separated list. Possible enum values are `fixed`,`variable`,`discount`,`tracker`
        filterbypaymenttype: `repayment` or `interestOnly`
        paginationlimit: Number of results per page
        sorttype: Adjusts results sorting. 
Possible values are: `initialRate`, `monthlyCost `, `totalApplicationFeesPounds`
        location: Filter by location. Possible values are: 
`england`, `northernIreland`, `scotland`, `wales`
        
    """
    url = f"https://uk-mortgage-search.p.rapidapi.com/"
    querystring = {'repaymentPeriod': repaymentperiod, 'propertyValue': propertyvalue, 'mortgageAmount': mortgageamount, }
    if page:
        querystring['page'] = page
    if filterbyinitialperiod:
        querystring['filterByInitialPeriod'] = filterbyinitialperiod
    if filterbyratetype:
        querystring['filterByRateType'] = filterbyratetype
    if filterbypaymenttype:
        querystring['filterByPaymentType'] = filterbypaymenttype
    if paginationlimit:
        querystring['paginationLimit'] = paginationlimit
    if sorttype:
        querystring['sortType'] = sorttype
    if location:
        querystring['location'] = location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uk-mortgage-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

