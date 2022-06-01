import requests
files = {
  "name": "Hold Alyssa Frye Harness boots 12R, Sz 7",
  "item_condition_id": 3,
  "category_name": "Women/Shoes/Boots",
}
print(requests.post("http://0.0.0.0:8080/v1/price", json=files).json())
