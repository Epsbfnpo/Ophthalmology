import os
import shutil

train_path="/raid5/weityx/Dataset/isic_saliency/train/"
saliency_path="/raid5/weityx/Dataset/isic_saliency/test/"


for i in range(7):
    file_path=os.path.join(saliency_path,str(i))
    for filename in os.listdir(file_path):
        shutil.move(os.path.join(file_path,filename),os.path.join(train_path,str(i),filename))



# saliency_train="/raid5/weityx/Dataset/isic_saliency/train/"
# with open('/raid5/weityx/Parametric-Contrastive-Learning/GPaCo/LT/dataset/data_txt/ISIC_val.txt', 'r') as f:
#     for row in f:
#         row = row.strip('\n').split()[0]
#         file_path=os.path.join(saliency_path,row)
#         if not os.path.exists(file_path):
#             past_row=row.replace("val","train")
#             # print(os.path.join(saliency_path,past_row))
#             # break
#             shutil.move(os.path.join(saliency_path,past_row),os.path.join(saliency_path,row))