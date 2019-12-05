import requests 
import glob
import os
import json

curr_files = glob.glob('../Data/310/L2/C1/*')
ref_files = glob.glob('../Data/1125/L2/C1/*')

curr_files = [os.path.abspath(x) for x in curr_files]
ref_files = [os.path.abspath(x) for x in ref_files]

ref_files = ref_files[:5]
curr_files = curr_files[:5]
os.environ['NO_PROXY']='localhost'
header = {
    'Content-Type': 'application/json',
}

data = {"curr_images":curr_files,"reference_images":ref_files}
data = json.dumps(data)

data2 = {"image":'/home/prajwal/Desktop/Design Lab Project/MSD_app/1_light_on/IMG_20191101_154652.jpg'}
data2 = json.dumps(data2)

response = requests.post('http://localhost:5000/serveMultiRequest', headers=header, data=data)

print(response.status_code)
print(response.json())