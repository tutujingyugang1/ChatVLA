import csv
import os
img_folder_path ='/home/jz08/zhouzy/data/vl_data/real-world/processed_ocr_val'
img_list = os.listdir(img_folder_path)
img_list.sort()
data = [['index','question','A','B','C','D','answer','category','abcLabel','image_path']]
for i, img_name in enumerate(img_list):
    data.append(
        [str(i), 'Solve the equation and select the card from bottom row with the correct answer on it.', '', '', '', '', '', '', '',
         img_name])
    # data.append([str(i), 'Solve the equation and point out the which card is the correct answer.','','','','','','','',img_name])

# Specify the file name
filename = "/home/jz08/LMUData/MMRO_mini.tsv"

# Writing to tsv file
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file, delimiter='\t')
    writer.writerows(data)

print(f"TSV file '{filename}' created successfully.")
