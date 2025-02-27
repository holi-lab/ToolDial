import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def magento_facebook_integration_extension_by_knowband(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Facebook is the most popular social media platform out there for us. So why not to simply buy and sell store products on Facebook? Easily bring your friends to shop online from your Facebook shop. The e-merchants can easily take the advantage of the Facebook customer base to boost their site traffic and engagement levels. 
		Knowband offers[ Magento FB Store extension](url=https://www.knowband.com/magento-fb-store), one perfect eCommerce solution for store merchants allowing them to integrate their online shop with the Facebook business page. Magento FB store integration module allows you to connect any number of fan pages and manage them easily from a single backend. Magento Facebook Shop extension provides one effective way of Facebook commerce which is convenient for both buyers and sellers. Look down and check out its various striking features.
		
		**Magento Facebook Integration Extension Features:**
		1. With Magento Facebook Store Setup, admin can connect any number of Facebook pages and manage them from a single backend interface.
		2. Magento Facebook Store Integration module offers easy synchronization of product listings on the Facebook Shop.
		3. The Google Analytics feature of Magento FB Shop extension allows admin to track the performance of the connected FB pages.
		4. The navigation menu of the Facebook store can be easily configured using Magento facebook shop integration module.
		5. Magento FB Store extension adds a search field on the Facebook shop allowing the customers to search easily for the products.
		6. Magento Facebook Integration extension allows the admin to add a logo image and banner image on the Homepage of Facebook Shop.
		7. The online Facebook Shop created using Magento Facebook Store Setup can be anytime enabled and disabled from the backend.
		8. The store admin can edit the homepage content of the Facebook Shop created using Magento Facebook Shop extension.
		9. The footer content of Facebook shop can also be configured from the backend of the Magento Facebook Store Integration module by Knowband.
		10. With Magento facebook integration module, admin also has the flexibility to decide the products to be displayed on the Facebook Store.
		11. Admin can change the theme color and font color for their Facebook shop using Magento FB Shop extension by Knowband.
		12. The number of type of featured products and featured categories can also be configured from the backend of Magento FB Store extension.
		
		**Admin Benefits of Magento FB store integration module:**
		1. With the large customer base of Facebook, admin can redirect the bulk amount of traffic and can boost the sales and conversions for their store.
		2. Magento Facebook store extension adds a 'Add to Cart' button on the Facebook Shop which redirects the customer to the main website.
		3. Magento FB shop integration module drives potential leads via Facebook fan page.
		4. Magento FB Shop extension increases product visibility on the most engaged social media platform like Facebook which contributes to store popularity among Facebook users.
		5. Magento Facebook Integration extension paves out a way to gain customer loyalty towards the store.
		
		**Related Links:**
		[User Manual](url=https://www.knowband.com/blog/user-manual/magento-fb-store-extension/)
		[Admin Demo](url=https://mademo5.knowband.com/fbstore/admin)
		[Front Demo](url=https://www.facebook.com/Stay-Connected-487520995007143/app/463084804200458/)"
    
    """
    url = f"https://magento-facebook-integration-extension-by-knowband.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "magento-facebook-integration-extension-by-knowband.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

