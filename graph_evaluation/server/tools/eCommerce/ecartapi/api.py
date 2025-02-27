import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getasinglecustomer(content_type: str, authorization: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Searches for the selected customer and shows detailed information.<br></p>
		<b>Important information:<br><b/>
			<b><ul>
			<li><a href = "https://docs.ecartapi.com/?version=latest#222f52a0-58f0-4cd9-bd11-68eb488a010b"> The costumer's ID must exist</a></li>
			</ul></b>"
    
    """
    url = f"https://ecartapi1.p.rapidapi.com/api/v2/customers/{is_id}"
    querystring = {'Content-Type': content_type, 'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecartapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getallproductvariants(authorization: str, productid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Searches for all the registered variants of a single product.<br></p>
		
		<b>Important information:<br><b/>
			<b><ul>
			<li><a href = "https://docs.ecartapi.com/?version=latest#6870b009-81d8-42c6-b910-3aaca8e5d0f2"> The product's ID must exist</a></li>
			</ul></b>
		
		<h3>FILTERS </h3>
		<table>
			<thead>
				<tr>
					<td>Filter</td>
					<td>Description</td>
				</tr>
			</thead>
			<tbody>
				<tr>
					<td><code>limit</code></td>
					<td>Limit size of the search.</td>
				</tr>
				<tr>
					<td><code>page</code></td>
					<td>Pagination number.</td>
				</tr>
				<tr>
					<td><code>sinceId</code></td>
					<td>Filter items since the Id given.</td>
				</tr>
				<tr>
					<td><code>offset</code></td>
					<td>Lower limit of the results block. Modifies the range of the results provided, range goes from offset to limit.</td>
				</tr>
			</tbody>
		</table>"
    
    """
    url = f"https://ecartapi1.p.rapidapi.com/api/v2/products/{productid}/variants"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecartapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsallorder_sfulfillments(content_type: str, authorization: str, orderid: str, offset: str='Integer [ 1, 2, 3 ... ]', page: str='[ token, 1 ]', limit: str='Integer [ 1, 100, 250 ]', sinceid: str='[ id ]', allstores: str='[ true, false ]', createdat_from: str='Date [ 2021-09-22T13:46:16-05:00 ]', createdat_to: str='Date [ 2021-09-22T13:46:16-05:00 ]', updatedat_to: str='Date [ 2021-09-22T13:46:16-05:00 ]', updatedat_from: str='Date [ 2021-09-22T13:46:16-05:00 ]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Shows all of the order's fulfillments detailed.<br></p>
		<b>Important information:<br><b/>
			<b><ul>
		<li>Filters are subject to ecommerce.</li>
			<li><a href = "https://docs.ecartapi.com/?version=latest#646f03ea-0962-4570-9a23-fa06320fc2e6"> The order's ID must exist</a></li>
			</ul></b>"
    offset: Lower limit of the results block. Modifies the range of the results provided, range goes from offset to limit

        page: Pagination number or token (Only when page object is available in response)
        limit: Limit size of the search
        sinceid: Filter items since the Id given
        allstores: Search items in all stores. ONLY Shiphero.
        createdat_from: Filter items by date created from. 
        createdat_to: Filter items by date created to.
        updatedat_to: Filter items by date updated to. 
        updatedat_from: Filter items by date updated from. 
        
    """
    url = f"https://ecartapi1.p.rapidapi.com/api/v2/orders/{orderid}/fulfillments"
    querystring = {'Content-Type': content_type, 'Authorization': authorization, }
    if offset:
        querystring['offset'] = offset
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    if sinceid:
        querystring['sinceId'] = sinceid
    if allstores:
        querystring['allStores'] = allstores
    if createdat_from:
        querystring['createdAt[from]'] = createdat_from
    if createdat_to:
        querystring['createdAt[to]'] = createdat_to
    if updatedat_to:
        querystring['updatedAt[to]'] = updatedat_to
    if updatedat_from:
        querystring['updatedAt[from]'] = updatedat_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecartapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getallcategories(authorization: str, siteid: str='String', limit: str='Int', productid: str='String', name: str='String', sinceid: str='String', page: str='String or number', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Searches for all the available categories.<br></p>
		
		<h3>FILTERS</h3>
		<table>
			<thead>
				<tr>
					<td>Filter</td>
					<td>Description</td>
				</tr>
			</thead>
			<tbody>
				<tr>
					<td><code>ids</code></td>
					<td>Filter categories by the ids given</td>
				</tr>
				<tr>
					<td><code>limit</code></td>
					<td>Limit size of the search</td>
				</tr>
		<tr>
					<td><code>page</code></td>
					<td>Pagination number.</td>
				</tr>
				<tr>
					<td><code>sinceId</code></td>
					<td>Filter categories since the Id given</td>
				</tr>
				<tr>
		      		<td><code>name</code></td>
		      		<td>Filter categories by name</td>
		    	</tr>
				<tr>
					<td><code>productId</code></td>
					<td>Filter categories by product id</td>
				</tr>
				<tr>
					<td><code>siteId</code></td>
					<td>Filter items by site id </td>
				</tr>
				<tr>
					<td><code>offset</code></td>
					<td>Lower limit of the results block. Modifies the range of the results provided, range goes from offset to limit</td>
				</tr>
			</tbody>
		</table>"
    siteid: Filter results based of the site id.
        limit: Default 50. Quantity of documents to be returned in the response.
        productid: Filter results based of products.
        name: Filter results based of the category name.
        sinceid: Filter results to be returned only after certain ID.
        page: Allows iteration through the results.
        
    """
    url = f"https://ecartapi1.p.rapidapi.com/api/v2/categories"
    querystring = {'Authorization': authorization, }
    if siteid:
        querystring['siteId'] = siteid
    if limit:
        querystring['limit'] = limit
    if productid:
        querystring['productId'] = productid
    if name:
        querystring['name'] = name
    if sinceid:
        querystring['sinceId'] = sinceid
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecartapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countallproducts(authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://ecartapi1.p.rapidapi.com/api/v2/products/count"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecartapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getasingleproductimage(authorization: str, is_id: str, productid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Searches for the selected image of a product.<br></p>
		
		<b>Important information:<br><b/>
			<b><ul>
			<li><a href = "https://docs.ecartapi.com/?version=latest#6870b009-81d8-42c6-b910-3aaca8e5d0f2"> The product's ID must exist</a></li>
			<li><a href= "https://docs.ecartapi.com/?version=latest#ff23601f-006f-4680-a5be-4607ff6bbc34">The product's image ID must exist.</a></li>
			</ul></b>"
    
    """
    url = f"https://ecartapi1.p.rapidapi.com/api/v2/products/{productid}/images/{is_id}"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecartapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def allproductlistings(authorization: str, limit: str='Int. Min: 0, Max: Subject to ecommerce.', createdat_from: str='YYYY-MM-DD', page: str='Number or hash.', createdat_to: str='YYYY-MM-DD', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    limit: Default 50. Quantity of documents to be returned in the response.
        createdat_from: Filter items by date of creation.
        page: Allows iteration through the results
        createdat_to: Filter items between their creation date range. createdAt[from]Required
        
    """
    url = f"https://ecartapi1.p.rapidapi.com/api/v2/products/listings"
    querystring = {'Authorization': authorization, }
    if limit:
        querystring['limit'] = limit
    if createdat_from:
        querystring['createdAt[from]'] = createdat_from
    if page:
        querystring['page'] = page
    if createdat_to:
        querystring['createdAt[to]'] = createdat_to
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecartapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getasingleorder(authorization: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Searches for a selected order and shows the detailed information.<br></p>
		<b>Important information:<br><b/>
			<b><ul>
			<li><a href = "https://docs.ecartapi.com/?version=latest#646f03ea-0962-4570-9a23-fa06320fc2e6"> The order's ID must exist</a></li>
			</ul></b>"
    
    """
    url = f"https://ecartapi1.p.rapidapi.com/api/v2/orders/{is_id}"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecartapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getallcustomers(content_type: str, authorization: str, email: str='String', updatedat_to: str='YYYY-MM-DD', name: str='String', ids: str='String', page: str='String or Number', createdat_to: str='YYYY-MM-DD', limit: str='Int', sinceid: str='String', updatedat_from: str='YYYY-MM-DD', createdat_from: str='YYYY-MM-DD', search: str='String', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<h3>FILTERS</h3>
		<table>
			<thead>
				<tr>
					<td>Filter</td>
					<td>Description</td>
				</tr>
			</thead>
			<tbody>
				<tr>
					<td><code>ids</code></td>
					<td>Filter cutomers by Id's provided</td>
				</tr>
				<tr>
					<td><code>limit</code></td>
					<td>Limit size of the search</td>
				</tr>
				<tr>
					<td><code>page</code></td>
					<td>Pagination number.</td>
				</tr>
				<tr>
					<td><code>sinceId</code></td>
					<td>Filter customers since the Id given</td>
				</tr>
				<tr>
					<td><code>createdAt[from]</code></td>
					<td>Filter customers by date created from. When used <code>createdAt[to]</code><i>Required</i></td>
				</tr>
				<tr>
		      		<td><code>createdAt[to]</code></td>
		      		<td>Filter customers by date created to. When used <code>createdAt[from]</code><i>Required</i></td>
		    	</tr>
				<tr>
					<td><code>updatedAt[from]</code></td>
					<td>Filter customers by date updated from. When used <code>updatedAt[to]</code><i>Required</i></td>
				</tr>
				<tr>
					<td><code>updatedAt[to]</code></td>
					<td>Filter customers by date updated to. When used <code>updatedAt[from]</code><i>Required</i></td>
				</tr>
				<tr>
		      		<td><code>name</code></td>
		      		<td>Filter categories by name</td>
		    	</tr>
				<tr>
		      		<td><code>search</code></td>
		      		<td>Search by string</td>
		    	</tr>
				<tr>
		      		<td><code>email</code></td>
		      		<td>Search customers by email</td>
		    	</tr>
				<tr>
					<td><code>offset</code></td>
					<td>Lower limit of the results block.
						Modifies the range of the results provided range goes from offset to limit</td>
				</tr>
			</tbody>
		</table>"
    email: Search based of email information.
        updatedat_to: Filter results between a date range. updatedAt[from]Required
        name: Filter results based of the customers name.
        ids: Returns the items inside of a list of ids separated by comas
        page: Allows iteration through the results.
        createdat_to: Filter results between their creation date range. createdAt[from]Required
        limit: Default 50. Quantity of documents to be returned in the response.
        sinceid: Filter results to be returned only after certain ID.
        updatedat_from: Filter results by date last update.
        createdat_from: Filter results by date of creation.
        search: General search on the resource.
        
    """
    url = f"https://ecartapi1.p.rapidapi.com/api/v2/customers"
    querystring = {'Content-Type': content_type, 'Authorization': authorization, }
    if email:
        querystring['email'] = email
    if updatedat_to:
        querystring['updatedAt[to]'] = updatedat_to
    if name:
        querystring['name'] = name
    if ids:
        querystring['ids'] = ids
    if page:
        querystring['page'] = page
    if createdat_to:
        querystring['createdAt[to]'] = createdat_to
    if limit:
        querystring['limit'] = limit
    if sinceid:
        querystring['sinceId'] = sinceid
    if updatedat_from:
        querystring['updatedAt[from]'] = updatedat_from
    if createdat_from:
        querystring['createdAt[from]'] = createdat_from
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecartapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getallorderfulfillments(authorization: str, status_status: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    status_status: Filters by orders fulfillment status
        
    """
    url = f"https://ecartapi1.p.rapidapi.com/api/v2/orders/services/fulfillment"
    querystring = {'Authorization': authorization, 'status[status]': status_status, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecartapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getallproducts(authorization: str, searchtype: str='String', status_active: str='Boolean.', sinceid: str='String.', createdat_to: str='YYYY-MM-DD', price_min: str='Number.', price_max: str='Number', createdat_from: str='YYYY-MM-DD', producttype: str='String.', sku: str='String.', sort: str='String. [ asc, desc ]', ids: str='String.', status_status: str='String', name: str='String.', categoryid: str='String.', status_visibility: str='Boolean.', updatedat_to: str='YYYY-MM-DD', page: str='String or Number.', sortby: str='String.', limit: str='String. Min: 0 Max: Subject to ecommerce.', updatedat_from: str='YYYY-MM-DD', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request used to obtain all the products in a store catalog.
		
		Find a general representation of all the filters and information EcartAPI can abet.
		
		#### For more detailed information please refer to our Supported Requests By Cart folder and find the eCommerce of your choice."
    searchtype: General search on the resource.
        status_active: Filter items by their inventory availability status.
        sinceid: Filter results to be returned only after certain ID.
        createdat_to: Filter items between their creation date range. createdAt[from]Required
        price_min: Filter items based on the minimum price.
        price_max: Filter items based on the max price.
        createdat_from: Filter items by date of creation.
        producttype: Filter results based of the product's type.
        sku: Filter results by sku.
        sort: Default desc. Order of the results returned.
        ids: Send a list of id's separated by commas.
        status_status: Filter items by the current status of the product.
        name: Filter results based of the item's name.
        categoryid: Filter results based of the category they are in.
        status_visibility: Filter items by their visibility status.
        updatedat_to: Filter items between a date range. updatedAt[from]Required
        page: Allows iteration through the results
        sortby: Default createdAt. The response will be sorted based on the key chosen.
        limit: Default 50. Quantity of documents to be returned in the response.
        updatedat_from: Filter items by date last update.
        
    """
    url = f"https://ecartapi1.p.rapidapi.com/api/v2/products"
    querystring = {'Authorization': authorization, }
    if searchtype:
        querystring['searchType'] = searchtype
    if status_active:
        querystring['status[active]'] = status_active
    if sinceid:
        querystring['sinceId'] = sinceid
    if createdat_to:
        querystring['createdAt[to]'] = createdat_to
    if price_min:
        querystring['price[min]'] = price_min
    if price_max:
        querystring['price[max]'] = price_max
    if createdat_from:
        querystring['createdAt[from]'] = createdat_from
    if producttype:
        querystring['productType'] = producttype
    if sku:
        querystring['sku'] = sku
    if sort:
        querystring['sort'] = sort
    if ids:
        querystring['ids'] = ids
    if status_status:
        querystring['status[status]'] = status_status
    if name:
        querystring['name'] = name
    if categoryid:
        querystring['categoryId'] = categoryid
    if status_visibility:
        querystring['status[visibility]'] = status_visibility
    if updatedat_to:
        querystring['updatedAt[to]'] = updatedat_to
    if page:
        querystring['page'] = page
    if sortby:
        querystring['sortBy'] = sortby
    if limit:
        querystring['limit'] = limit
    if updatedat_from:
        querystring['updatedAt[from]'] = updatedat_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecartapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getasingleproduct(authorization: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Searches for a selected product and shows the detailed information.<br></p>
		<b>Important information:<br><b/>
			<b><ul>
			<li><a href = "https://docs.ecartapi.com/?version=latest#6870b009-81d8-42c6-b910-3aaca8e5d0f2"> The product's ID must exist</a></li>
			</ul></b>"
    
    """
    url = f"https://ecartapi1.p.rapidapi.com/api/v2/products/{is_id}"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecartapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countallcostumers(authorization: str, content_type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://ecartapi1.p.rapidapi.com/api/v2/customers/count"
    querystring = {'Authorization': authorization, 'Content-Type': content_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecartapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countallproductvariants(authorization: str, productid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Counts all the registered variants of a single product.<br></p>
		
		<b>Important information:<br><b/>
			<b><ul>
			<li><a href = "https://docs.ecartapi.com/?version=latest#6870b009-81d8-42c6-b910-3aaca8e5d0f2"> The product's ID must exist</a></li>
			</ul></b>"
    
    """
    url = f"https://ecartapi1.p.rapidapi.com/api/v2/products/{productid}/variants/count"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecartapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsingleproductvariant(authorization: str, is_id: str, productid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Searches for the selected variant and shows the detailed information.<br></p>
		
		<b>Important information:<br><b/>
			<b><ul>
			<li><a href = "https://docs.ecartapi.com/?version=latest#6870b009-81d8-42c6-b910-3aaca8e5d0f2"> The product's ID must exist</a></li>
			<li><a href = "https://docs.ecartapi.com/?version=latest#5e6aa8c0-1d39-4a0e-8d8c-a2f2d294528a"> The product's variant ID must exist.</a></li>
			</ul></b>"
    
    """
    url = f"https://ecartapi1.p.rapidapi.com/api/v2/products/{productid}/variants/{is_id}"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecartapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcarrierservices(authorization: str, content_type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://ecartapi1.p.rapidapi.com/api/v2/services/carriers"
    querystring = {'Authorization': authorization, 'Content-Type': content_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecartapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getallproduct_simages(authorization: str, productid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Searches for all the images of a single product.<br></p>
		
		<b>Important information:<br><b/>
			<b><ul>
			<li><a href = "https://docs.ecartapi.com/?version=latest#6870b009-81d8-42c6-b910-3aaca8e5d0f2"> The product's ID must exist</a></li>
			</ul></b>
			
		<h3>FILTERS</h3>
		<table>
			<thead>
				<tr>
					<td>Filter</td>
					<td>Description</td>
				</tr>
			</thead>
			<tbody>
		<tr>
					<td><code>sinceId</code></td>
					<td>Filter items since the Id given</td>
				</tr>
		</tbody>
		</table>"
    
    """
    url = f"https://ecartapi1.p.rapidapi.com/api/v2/products/{productid}/images"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecartapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsasingleorderfulfillment(authorization: str, content_type: str, orderid: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Shows a selected order fulfillment with detail.<br></p>
		<b>Important information:<br><b/>
			<b><ul>
			<li><a href = "https://docs.ecartapi.com/?version=latest#646f03ea-0962-4570-9a23-fa06320fc2e6"> The order's ID must exist</a></li>
			<li><a href = "https://docs.ecartapi.com/?version=latest#043a96c7-cf6d-4dd7-8f3e-76dba5a8a2d1"> The fulfillment order's ID must exist</a></li>
			</ul></b>"
    
    """
    url = f"https://ecartapi1.p.rapidapi.com/api/v2/orders/{orderid}/fulfillments/{is_id}"
    querystring = {'Authorization': authorization, 'Content-Type': content_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecartapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countallcategories(authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://ecartapi1.p.rapidapi.com/api/v2/categories/count"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecartapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsinglecarrierservices(authorization: str, content_type: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://ecartapi1.p.rapidapi.com/api/v1/services/carriers/{is_id}"
    querystring = {'Authorization': authorization, 'Content-Type': content_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecartapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getasinglecategory(authorization: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Searches for the selected category and shows detailed information.<br></p>
		<b>Important information:<br><b/>
			<b><ul>
			<li><a href = "https://docs.ecartapi.com/?version=latest#aa4a4c0b-6ab3-49e8-b7e3-931c8226af66"> The categories ID must exist</a></li>
			</ul></b>"
    
    """
    url = f"https://ecartapi1.p.rapidapi.com/api/v2/categories/{is_id}"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecartapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getyourstoreinformation(authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access store information fast with the access token we provided."
    
    """
    url = f"https://ecartapi1.p.rapidapi.com/accesses/me"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecartapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def infoproductlisting(authorization: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://ecartapi1.p.rapidapi.com/api/v2/products/listings/{is_id}"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecartapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getallorders(authorization: str, search: str='String. [ customer@email.com ]', sinceid: str='String', page: str='String or Number', status_financial: str='Values subject to ecommerce', status_status: str='Values subject to ecommerce', draft: str='[ true, false ]', status_ecartapi: str='[ pending, authorized, paid, partially_paid, refunded, partially_refunded, cancelled, delete, shipped, other ]', createdat_to: str='YYYY-MM-DD', ids: str='String', recent: str='[ true, false ]', fulfillmentstatus: str='Values subject to ecommerce', offset: str='Int', createdat_from: str='YYYY-MM-DD', updatedat_to: str='YYYY-MM-DD', limit: str='Int', status_id: str='Values subject to ecommerce', allstores: str='[true, false]', sort: str='[ asc, desc ]', updatedat_from: str='YYYY-MM-DD', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p><b>Note:</b><br>
		Filters are available depending of the ecommerce
		</p>"
    search: Search by key-word
        sinceid: Request orders since a certain Id
        page: Pagination is numbered or by token(only when response includes page object)
        status_financial: Filter orders by financial status. Values subject to ecommerce
        status_status: Filter orders by status. Values subject to ecommerce
        draft: Shopify Only.
        status_ecartapi: Filter orders by status.
        createdat_to: Filter orders by date created to.
        ids: List of order ids separated by comas
        recent: Shows the most recent orders, including the ones with dates before e expiration_date and have not been qualified
        fulfillmentstatus: Filter orders by the fulfillment status. Values subject to ecommerce
        offset: Lower limit of the results block. Modifies the range of the results provided, range goes from offset to limit
        createdat_from: Filter orders by date created from.
        updatedat_to: Filter orders by date updated to.
        limit: Limit of documents requested by call
        status_id: Filter orders by status Id. Values are subject to ecommerce
        allstores: Search items in all stores.
        sort: Shows orders by ascending or descending dates.
        updatedat_from: Filter orders by date updatedfrom.
        
    """
    url = f"https://ecartapi1.p.rapidapi.com/api/v2/orders"
    querystring = {'Authorization': authorization, }
    if search:
        querystring['search'] = search
    if sinceid:
        querystring['sinceId'] = sinceid
    if page:
        querystring['page'] = page
    if status_financial:
        querystring['status[financial]'] = status_financial
    if status_status:
        querystring['status[status]'] = status_status
    if draft:
        querystring['draft'] = draft
    if status_ecartapi:
        querystring['status[ecartapi]'] = status_ecartapi
    if createdat_to:
        querystring['createdAt[to]'] = createdat_to
    if ids:
        querystring['ids'] = ids
    if recent:
        querystring['recent'] = recent
    if fulfillmentstatus:
        querystring['fulfillmentStatus'] = fulfillmentstatus
    if offset:
        querystring['offset'] = offset
    if createdat_from:
        querystring['createdAt[from]'] = createdat_from
    if updatedat_to:
        querystring['updatedAt[to]'] = updatedat_to
    if limit:
        querystring['limit'] = limit
    if status_id:
        querystring['status[id]'] = status_id
    if allstores:
        querystring['allStores'] = allstores
    if sort:
        querystring['sort'] = sort
    if updatedat_from:
        querystring['updatedAt[from]'] = updatedat_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecartapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getfulfillmentservice(authorization: str, is_id: str, ecommerce: str='true', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://ecartapi1.p.rapidapi.com/api/v2/services/fulfillment/{is_id}"
    querystring = {'Authorization': authorization, }
    if ecommerce:
        querystring['ecommerce'] = ecommerce
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecartapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getallfulfillmentservices(authorization: str, ecommerce: str='true', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://ecartapi1.p.rapidapi.com/api/v2/services/fulfillment"
    querystring = {'Authorization': authorization, }
    if ecommerce:
        querystring['ecommerce'] = ecommerce
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecartapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countallorders(authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://ecartapi1.p.rapidapi.com/api/v2/orders/count"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecartapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getasinglewebhoook(authorization: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://ecartapi1.p.rapidapi.com/api/v2/webhooks/{is_id}"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecartapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getallstorewebhoooks(authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://ecartapi1.p.rapidapi.com/api/v2/webhooks"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecartapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getallcarts(authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://ecartapi1.p.rapidapi.com/api/v2/carts"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecartapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsinglecart(authorization: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://ecartapi1.p.rapidapi.com/api/v2/carts/{is_id}"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecartapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

