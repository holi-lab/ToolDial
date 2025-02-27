import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def zatec(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Zatec là một doanh nghiệp ra đời với mong muốn trở thành một giải pháp toàn diện cho người tiêu dùng về các sản phẩm kệ để hàng bao gồm: Kệ sắt, kệ siêu thị, Kệ sắt V lỗ, Kệ trung tải, kệ sắt, kệ để hàng, kệ đa năng, kệ đa năng mặt lưới, kệ sắt lắp ráp, kệ sắt 3 tầng,...
		Zatec đã và đang là nhà thầu phụ uy tín, cung cấp các giải pháp về kho hàng cho các nhà máy, nhà xưởng trong các khu công nghiệp, cũng như hệ thống siêu thị, cửa hàng tạp hóa lớn và nhỏ trên khắp địa bàn Hà Nội, Hồ Chí Minh và 63 tỉnh thành.
		#Zatec  #kệ_sắt_v_lô #kệ_trung_tải #kệ_siêu_thị #báo_giá #bảng_giá 
		Địa chỉ : 09, LK 06, Dọc Bún 1, KĐT Văn Khê, Hà Đông, Hà Nội 
		Warehouse: 07, Lê Đức Thọ, Nam Từ Liêm, Hà Nội
		Hotline 0963132714
		Gmail: sales.zatec@gmail.com
		Google map https://www.google.com/maps?cid=6954165448449146962 
		Website https://www.zatec.vn/ 
		https://www.zatec.vn/ke-sat-v-lo 
		https://www.zatec.vn/ke-trung-tai 
		https://www.zatec.vn/gia-ke-sieu-thi-re-nhat-hien-nay 
		https://www.zatec.vn/ke-da-nang-mat-luoi 
		https://www.facebook.com/zatecvn 
		https://sites.google.com/view/zatec/home
		https://www.blogger.com/profile/18005754094756663739 
		https://gallery.autodesk.com/users/Y5E2MQ74KZS9WNT6 
		https://www.reverbnation.com/artist/zatec
		https://bitbucket.org/zatec/workspace/snippets/5koMbL
		https://independent.academia.edu/zatec
		https://www.figma.com/@zatec
		https://connect.garmin.com/modern/profile/42073ab4-e908-43cd-803a-a5ac909471a4
		https://gallery.autodesk.com/users/Y5E2MQ74KZS9WNT6
		https://fliphtml5.com/homepage/gapdg
		https://www.producthunt.com/@zatec
		https://ko-fi.com/za_tec
		https://onlyfans.com/zatec
		https://padlet.com/seozatec/zatec
		https://sketchfab.com/zatec
		https://www.mojomarketplace.com/user/zatec-7c962sKipu 
		https://comicvine.gamespot.com/profile/zatec/ 
		https://www.beatstars.com/zatec/about"
    
    """
    url = f"https://zatec.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zatec.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

