import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# 解析XML文件
tree = ET.parse('D:/Pyproject/StarPerf_Simulator/config/ground_stations/Starlink.xml')
root = tree.getroot()

# 存储地理位置信息
locations = []

# 遍历每个GS节点
for gs_node in root:
    latitude = float(gs_node.find('Latitude').text)
    longitude = float(gs_node.find('Longitude').text)
    description = gs_node.find('Description').text
    locations.append((latitude, longitude, description))

# 绘制地理位置
fig, ax = plt.subplots()
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('GS Locations')

for location in locations:
    latitude, longitude, description = location
    ax.plot(longitude, latitude, 'ro')
    ax.text(longitude, latitude, description, ha='right')

plt.savefig("D:/Pyproject/StarPerf_Simulator/config/ground_stations/GS_location.jpg")
plt.show()
