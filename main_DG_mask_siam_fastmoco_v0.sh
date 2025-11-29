
# APTOS DEEPDR FGADR IDRID MESSIDOR RLDR


# CUDA_VISIBLE_DEVICES=$gpu python main_v0.py  --root YOUR_DATASET_ROOT_PATH
#                --algorithm DRGen
#                --dg_mode DG
#                --source-domains DATASET2 DATASET3 ...
#                --target-domains DATASET1 
#                --output YOUR_OUTPUT_DIR
# model=$1
# lr=$2
# mask_ratio=$3
# block_size=$4
# sup=$5
# fastcoco=$6
# masked=$7
# kd=$8
# batch_size=$9
# epochs=${10}
# gpu=${11}

model=$1
lr=$2
mask_ratio=$3
block_size=$4
sup=$5
fastcoco=$6
masked=$7
kd=$8
k=$9
p=${10}
batch_size=${11}
epochs=${12}
gpu=${13}


CUDA_VISIBLE_DEVICES=$gpu python main_v0.py  --root ./GDRBench/ \
                --algorithm GDRNet_MASK_SIAM \
                --dg_mode DG \
                --source-domains DEEPDR FGADR IDRID MESSIDOR RLDR \
                --target-domains APTOS  \
                --output ./DG/GDRNet_Mask_SIAM_FASTMOCO_V0/${model}_${mask_ratio}_${block_size}_${lr}_${sup}_${fastcoco}_${masked}_${kd}_${k}_${p}_${batch_size}_${epochs}/APTOS \
                --learning_rate $lr \
                --backbone ${model} \
                --mask_ratio $3 \
                --block_size $4 \
                --fastmoco $fastcoco \
                --masked $masked \
                --batch_size $batch_size \
                --epochs $epochs \
                --sup $sup \
                --kd $kd


CUDA_VISIBLE_DEVICES=$gpu python main_v0.py  --root ./GDRBench/ \
                --algorithm GDRNet_MASK_SIAM \
                --dg_mode DG \
                --source-domains APTOS FGADR IDRID MESSIDOR RLDR \
                --target-domains DEEPDR  \
                --output ./DG/GDRNet_Mask_SIAM_FASTMOCO_V0/${model}_${mask_ratio}_${block_size}_${lr}_${sup}_${fastcoco}_${masked}_${kd}_${k}_${p}_${batch_size}_${epochs}/DEEPDR \
                --learning_rate $lr \
                --backbone ${model} \
                --mask_ratio $3 \
                --block_size $4 \
                --fastmoco $fastcoco \
                --masked $masked \
                --batch_size $batch_size \
                --epochs $epochs \
                --sup $sup \
                --kd $kd


CUDA_VISIBLE_DEVICES=$gpu python main_v0.py  --root ./GDRBench/ \
                --algorithm GDRNet_MASK_SIAM \
                --dg_mode DG \
                --source-domains APTOS DEEPDR IDRID MESSIDOR RLDR \
                --target-domains FGADR  \
                --output ./DG/GDRNet_Mask_SIAM_FASTMOCO_V0/${model}_${mask_ratio}_${block_size}_${lr}_${sup}_${fastcoco}_${masked}_${kd}_${k}_${p}_${batch_size}_${epochs}/FGADR \
                --learning_rate $lr \
                --backbone ${model} \
                --mask_ratio $3 \
                --block_size $4 \
                --fastmoco $fastcoco \
                --masked $masked \
                --batch_size $batch_size \
                --epochs $epochs \
                --sup $sup \
                --kd $kd


CUDA_VISIBLE_DEVICES=$gpu python main_v0.py  --root ./GDRBench/ \
                --algorithm GDRNet_MASK_SIAM \
                --dg_mode DG \
                --source-domains APTOS DEEPDR FGADR MESSIDOR RLDR \
                --target-domains IDRID  \
                --output ./DG/GDRNet_Mask_SIAM_FASTMOCO_V0/${model}_${mask_ratio}_${block_size}_${lr}_${sup}_${fastcoco}_${masked}_${kd}_${k}_${p}_${batch_size}_${epochs}/IDRID \
                --learning_rate $lr \
                --backbone ${model} \
                --mask_ratio $3 \
                --block_size $4 \
                --fastmoco $fastcoco \
                --masked $masked \
                --batch_size $batch_size \
                --epochs $epochs \
                --sup $sup \
                --kd $kd


CUDA_VISIBLE_DEVICES=$gpu python main_v0.py  --root ./GDRBench/ \
                --algorithm GDRNet_MASK_SIAM \
                --dg_mode DG \
                --source-domains APTOS DEEPDR FGADR IDRID RLDR \
                --target-domains MESSIDOR  \
                --output ./DG/GDRNet_Mask_SIAM_FASTMOCO_V0/${model}_${mask_ratio}_${block_size}_${lr}_${sup}_${fastcoco}_${masked}_${kd}_${k}_${p}_${batch_size}_${epochs}/MESSIDOR \
                --learning_rate $lr \
                --backbone ${model} \
                --mask_ratio $3 \
                --block_size $4 \
                --fastmoco $fastcoco \
                --masked $masked \
                --batch_size $batch_size \
                --epochs $epochs \
                --sup $sup \
                --kd $kd


CUDA_VISIBLE_DEVICES=$gpu python main_v0.py  --root ./GDRBench/ \
                --algorithm GDRNet_MASK_SIAM \
                --dg_mode DG \
                --source-domains APTOS DEEPDR FGADR IDRID MESSIDOR \
                --target-domains RLDR  \
                --output ./DG/GDRNet_Mask_SIAM_FASTMOCO_V0/${model}_${mask_ratio}_${block_size}_${lr}_${sup}_${fastcoco}_${masked}_${kd}_${k}_${p}_${batch_size}_${epochs}/RLDR \
                --learning_rate $lr \
                --backbone ${model} \
                --mask_ratio $3 \
                --block_size $4 \
                --fastmoco $fastcoco \
                --masked $masked \
                --batch_size $batch_size \
                --epochs $epochs \
                --sup $sup \
                --kd $kd
