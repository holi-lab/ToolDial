import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def knowband_s_magneto_2_multi_vendor_marketplace_a_smart_solution_for_e_store_owners(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Nowadays store owners are looking for ways to extend the store reach and enhance their sales. Having a mobile app and building a Multi-vendor store is one of the prominent ways to catch customer attention and increase the customer return rate. If you are an online store owner all you need is to do is to install the[ Multi-seller Marketplace extension](url=https://www.knowband.com/magento-2-multi-vendor-marketplace) by Knowband. Within 3 simple clicks, your marketplace is ready to sell goods of multiple third-party sellers. Multiple vendors can register their store after submitting a simple registration form. Once the store owner will approve the account request, sellers can start listing the products and updating their shop profile which includes logo, banners, shop name etc.
		
		What makes the Magento 2 Multi-seller Marketplace by Knowband different?
		1.No coding /technical Knowledge
		2.Free installation and configuration
		3.3 simple click setup
		4.3 months of support
		5.Mobile app compatibility
		6.Affordable cost
		
		Main Highlights of Magento 2 Marketplace module are as under:
		
		Sellers can register their store by filling a simple registration form.
		Store owner can charge global and category commission on the sellers. The amount of commission admin charges the seller is pre-set from the admin panel. 
		Store owner can charge fixed or percentage commission on sellers.
		Customers can also get themselves registered as Sellers.
		The seller can receive low stock notifications about the products low in the count.
		Multi-seller Marketplace module offers all product type compatibility.
		Membership plan of Magneto 2 Marketplace is an additional source of income. In the plan, the admin can decide the validity of product sale.
		Vacation mode for sellers to close their shop for a temporary period. 
		All the orders placed at the Marketplace gets listed at the Order tab of the Marketplace.
		Store owners update the order status of orders from the backend. If the order handling feature is enabled, sellers can update the order status from seller panel.
		Customers can easily contact the sellers from the seller portal. For each query or contact, a ticket is generated. These tickets help the admin to keep a count of the queries and their status.
		Customers are free to submit their feedback about products and sellers from the front end of the store.
		All the reviews are directly monitored by the admin.
		
		For additional information, refer to the Complete [https://www.knowband.com/blog/user-manual/magento-2-multi-vendor-marketplace-user-manual/](url=https://www.knowband.com/blog/user-manual/magento-2-multi-vendor-marketplace-user-manual/)
		
		Watch the working of Magento 2 Marketplace module [Video](url=https://youtu.be/-jRfj9mUTVI)
		
		In case of query or discussion about the Magento 2 Marketplace module, contact us support@Knowband.com"
    
    """
    url = f"https://knowband-s-magneto-2-multi-vendor-marketplace-a-smart-solution-for-e-store-owners.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "knowband-s-magneto-2-multi-vendor-marketplace-a-smart-solution-for-e-store-owners.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

