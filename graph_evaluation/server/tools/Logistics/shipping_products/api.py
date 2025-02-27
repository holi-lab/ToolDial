import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def retrieve_a_list_of_shipping_products(from_country: str, height_unit: str='cm', length: int=100, width_unit: str='cm', weight: int=1500, width: int=50, carrier: str='postnl', to_country: str='DE', height: int=10, weight_unit: str='gram', contract_pricing: bool=None, lead_time_hours: int=24, length_unit: str='cm', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve a list of shipping methods that are associated with your default sender address, filtered by specific criteria such as parcel dimensions, weight classes, from and to country and **shipping functionality.** 
		
		In situations where you need to find a method which supports a specific means of delivery, type of parcel or delivery deadline, for example, this endpoint allows you to filter all available shipping methods based on one or more query parameters. 
		
		The response body will include the `id` of any suitable methods, which you can then use to <a href="Create-a-parcel">**Create a parcel**</a> and announce the shipment directly. 
		
		<!-- theme: warning -->
		> You must have either <a href="Getting-started">**enabled a carrier**</a> in your Sendcloud account, or connected your own direct <a href="contracts">**carrier contract**</a>, in order to be able to retrieve shipping methods related to that carrier via this endpoint. 
		
		
		<!-- theme: info -->
		
		>To filter by `{shipping_functionality}`, you can find a glossary of accepted values and a description of each functionality under the <a href="Get-shipping-functionalities">**Get shipping functionalities**</a> endpoint. 
		---
		### Use cases
		Find a shipping method `id` for a parcel weighing 5kg, shipping from the Netherlands to the United States with carrier PostNL. 
		
		<span style="background-color: #1D97FF">GET</span> `https://panel.sendcloud.sc/api/v2/shipping-products?from_country=NL&to_country=US&carrier=postnl&weight=5&weight_unit=kilogram`
		
		Find a shipping method which includes the Age Check functionality for shipping alcohol products from France to Belgium. 
		
		<span style="background-color: #1D97FF">GET</span> `https://panel.sendcloud.sc/api/v2/shipping-products?from_country=NL&to_country=BE&age_check=18`
		
		Find a shipping method that supports same day delivery for a product shipping from the Netherlands to the Netherlands. 
		
		<span style="background-color: #1D97FF">GET</span> `https://panel.sendcloud.sc/api/v2/shipping-products?from_country=NL&to_country=NL&delivery_deadline=sameday`
		
		Find a **return** shipping method for a parcel returning from France to the Netherlands:
		
		<span style="background-color: #1D97FF">GET</span> `https://panel.sendcloud.sc/api/v2/shipping-products?from_country=FR&to_country=NL&returns=true`"
    from_country: A country ISO 2 code for a from country (origin country).
        height_unit: The unit for the shipment height. Required if the “height” parameter is provided.
        length: The length of the shipment. Required if the “width” and/or “height” parameters are provided.
        width_unit: The unit for the shipment width. Required if the “width” parameter is provided.
        weight: The weight of the shipment.
        width: The width of the shipment. Required if the “length” and/or “height” parameters are provided.
        carrier: A carrier code.
        to_country: A country ISO 2 code for a to country (destination country). Required if the “contract_pricing” parameter is provided.
        height: The height of the shipment. Required if the “width” and/or “length” parameters are provided.
        weight_unit: The unit for the shipment weight. Required if the “weight” parameter is provided.
        contract_pricing: Whether to include contract prices in the response. This parameter is only available to users upon request. Please contact customer support.
        lead_time_hours: Filters shipping products (rather its associated shipping methods) by their transit time. This is the estimated time a shipment takes between a parcel is being picked up by a carrier and it reaching its destination. This parameter supports comparison operators (e.g. `lead_time_hours[gte]=12` for filtering out shipments that have less than 12 hours transit time or `lead_time_hours[gte=24]&lead_time_hours[lte]=48` for only including shipping products whose methods have a transit time between 24 and 48 hours). Available operators are “eq” (equal), “gt” (greater than), “lt” (less than), “gte” (greater than or equal), and “lte” (less than or equal).
        length_unit: The unit for the shipment length. Required if the “length” parameter is provided.
        
    """
    url = f"https://shipping-products.p.rapidapi.com/shipping-products"
    querystring = {'from_country': from_country, }
    if height_unit:
        querystring['height_unit'] = height_unit
    if length:
        querystring['length'] = length
    if width_unit:
        querystring['width_unit'] = width_unit
    if weight:
        querystring['weight'] = weight
    if width:
        querystring['width'] = width
    if carrier:
        querystring['carrier'] = carrier
    if to_country:
        querystring['to_country'] = to_country
    if height:
        querystring['height'] = height
    if weight_unit:
        querystring['weight_unit'] = weight_unit
    if contract_pricing:
        querystring['contract_pricing'] = contract_pricing
    if lead_time_hours:
        querystring['lead_time_hours'] = lead_time_hours
    if length_unit:
        querystring['length_unit'] = length_unit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shipping-products.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_a_list_of_shipping_functionalities(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists all of the available **shipping functionalities** across the Sendcloud system. A shipping functionality is an additional characteristic or 'add on' service that defines one shipping method from another. Some functionalities are related to the form of the shipment that's accepted for shipment, such as letterbox, parcel or pallet, while other functionalities specify any additional handling that's required, such as Age check or Signature required. Functionalities can also denote specific delivery deadlines or weekend delivery availability. 
		
		<!-- theme: info -->
		
		>The below **Glossary of shipping functionalities** provides a description of every possible shipping functionality associated with a Sendcloud-supported shipping method.
		
		Shipping functionality             | Description
		------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
		`age_check`             | Indicates whether the recipient must be above a certain age (e.g., to accept alcohol products).
		`b2b`                   | Indicates whether the shipment is a b2b shipment.
		`b2c`                   | Indicates whether the shipment is a b2c shipment.
		`boxable`               | Indicates whether the shipment fits in a box.
		`bulky_goods`           | Indicates whether the shipment is bulky, e.g. it does not fit in a box.
		`carrier_billing_type`  | Indicates whether the shipment is billed on a country to country basis, or based on a shipping zone to shipping zone. Example is shipping from mainland Spain to the Canary Islands (a zone) or Netherlands to the Netherlands (country to country).
		`cash_on_delivery`      | Indicates whether the receiver of the shipment should pay for the shipment when receiving it.
		`dangerous_goods`       | Indicates whether the shipment can contain dangerous goods.
		`delivery_attempts`     | Indicates the number of delivery attempts the carrier should attempt before returning (or discarding) the parcel.
		`delivery_before`       | Indicates whether shipment will be delivered before a certain time of the day (Example: before 12:00).
		`delivery_deadline`     | Indicates the period of time in which the shipment will be delivered (Example: 24 hours, 28 hours).
		`direct_contract_only`  | Indicates whether shipping is only possible using your own carrier contract, or if it is possible using Sendcloud contract rates.
		`eco_delivery`          | Indicates whether the shipping process will be environmently friendly.
		`ers`                   | Indicates whether shipment will use the ERS (Easy Return Solution) system.
		`first_mile`            | Indicates how transportation of the first mile of the shipment will take place. Example is that the parcel is dropped of at a service point, or is picked up by the carrier.
		`flex_delivery`         | Indicates whether the receiver of the parcel can, before delivery takes place, choose where and when the shipment should be delivered.
		`form_factor`           | Indicates the form factor of the parcel. Examples are letter, pallet, parcel.
		`fragile_goods`         | Indicates whether shipment can contain fragile goods (glass, electronics, etcetera).
		`fresh_goods`           | Indicates whether shipment can contain fresh goods (e.g. food with an expiration date).
		`harmonized_label`      | Indicates whether shipment label contains the customs information on as well.
		`id_check`              | Indicates whether the receiver should identify him/herself to the carrier driver, conforming the identity of the receiver.
		`incoterm`              | Indicates what incoterm is used for an (international) shipment. Mainly used to determine if the receiver or the sender pays the customs duties.
		`insurance`             | Indicates whether the shipment has carrier insurance or not.
		`labelless`             | Indicates whether a return shipment can be done using only a QR code or numerical number, needed by the end-consumer to return the parcel. In other words, no shipping label is required.
		`last_mile`             | Indicates what the last mile of the shipment looks like. For instance, the shipment can be delivered to a service point or a home address.
		`manually`              | Indicates a subset of Deutsche Post shipping methods where a consumer should manually attach the label to the parcel.
		`multicollo`            | Indicates whether the shipment can be a multi-collo shipment. Note: Not all carriers support multicollo shipment. See the supported carriers and more in our help center
		`neighbor_delivery`     | Indicates whether shipment is allowed to be delivered at the neighbours of the receiver.
		`non_conveyable`        | Indicates whether the shipment fits on a conveyor belt.
		`personalized_delivery` | Indicates a subset of Deutsche Post shipping methods shipping to a consumer.
		`premium`               | Indicates whether the carrier identifies the shipments shipping process as premium.
		`priority`              | Indicates the priority level of the shipment. Examples are Express or Standard.
		`registered_delivery`   | Indicates whether a Proof of Delivery (POD) is communicated to the sender.
		`returns`               | Indicates whether the shipment should be a return shipment or not.
		`segment`               | Indicates the international pricing zone for PostNL shipments.
		`service_area`          | Indicates the service area of the shipment. Examples are domestic or international.
		`signature`             | Indicates whether the shipment requires a signature upon delivery.
		`size`                  | Indicates the allowed size of the shipment.
		`sorted`                | Indicates whether the shipment(s) are handed over the carrier in a sorted fashion, decreasing costs.
		`surcharge`             | Indicates whether the carrier can surcharge the shipment later, based on (volumetric) weight.
		`tracked`               | Indicates whether the shipment can be tracked online.
		`tyres`                 | Indicates whether the shipment can be used to ship tyres.
		`weekend_delivery`      | Indicates whether shipment can delivery in the weekend or on a specific weekend-day."
    
    """
    url = f"https://shipping-products.p.rapidapi.com/shipping-functionalities"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shipping-products.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

