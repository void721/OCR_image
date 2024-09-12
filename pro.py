

import os
import json
import base64
from PIL import Image
from io import BytesIO

# 假设的处理函数，输入图片路径，返回处理结果
def process_image(image_path):
    import easyocr
    reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory
    result = reader.readtext(image_path,detail=0)
    # print(result)
    all_res=""
    for i in range(len(result)):
        all_res+=result[i]
        all_res+=" "
    return all_res

# 遍历文件夹并处理图片
def process_images_in_folder(folder_path, output_json_path):
    images_data = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            print(f"processing image_path {image_path}")

            # 读取图片并转换为base64编码
            # with open(image_path, "rb") as image_file:
            #     image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

            # 调用处理函数获取内容
            content = process_image(image_path)

            # 将图片和内容添加到数据列表
            images_data.append({
                "image": image_path.split("/")[-1],
                "content": content
            })

    # 将数据写入JSON文件
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(images_data, json_file, ensure_ascii=False, indent=4)

# 指定文件夹路径和输出JSON路径
folder_path = '/home/zyc/proj/OCR_image/OCRbench'
output_json_path = 'OCR_bench.json'

# 处理文件夹中的图片并生成JSON
process_images_in_folder(folder_path, output_json_path)
