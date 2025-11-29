
# method=$1

# for source in  APTOS DEEPDR FGADR IDRID MESSIDOR RLDR

# do 
#     python main_v0_test.py --root ./GDRBench/ \
#                 --algorithm GDRNet_MASK_SIAM \
#                 --dg_mode DG \
#                 --source-domains ${source} \
#                 --target-domains  ${target} \
#                 --output ./best_DG/${source}
# done
                # --target-domains DEEPDR FGADR IDRID MESSIDOR RLDR \

# for source in  APTOS DEEPDR FGADR IDRID MESSIDOR RLDR


# CUDA_VISIBLE_DEVICES=2 python main_v0_test.py --root ./GDRBench/ --algorithm GDRNet_MASK_SIAM_SIAM  --dg_mode DG  --source-domain RLDR --target-domains APTOS DEEPDR FGADR IDRID MESSIDOR DDR EYEPACS  --output ./best_DG/QT_Dual/ESDG/resnet50_resnet50_0.5_32_0.001/RLDR   --backbone resnet50  --mask_ratio 0.5  --block_size 32 --learning_rate 0.001


# CUDA_VISIBLE_DEVICES=3 python main_v0_test.py --root ./GDRBench/ --algorithm GDRNet_MASK_SIAM  --dg_mode DG  --source-domain RLDR --target-domains APTOS DEEPDR FGADR IDRID MESSIDOR DDR EYEPACS  --output ./best_DG/QT_Dual/ESDG/resnet50_resnet50_0.5_32_0.001/RLDR   --backbone resnet50  --mask_ratio 0.5  --block_size 32 --learning_rate 0.001 --fastmoco $fastcoco




# bash main_ESDG_mask_siam_fastmoco.sh resnet50 0.001 0.5 32 0.0 16 4

# bash main_DG_mask_siam_fastmoco_v0_test.sh resnet50 0.0005 0.5 32 0.6 1.5 0.5 0.03 32 100 0  DDR  # 

# 效果最好的模型/home/user/Projects/Medical-Image/Ophthalmology/DGDR/result/fundusaug/best_DG/resnet50_0.5_32_0.00045_0.6_1.5_0.5_0.1_32_100

model=$1
lr=$2
mask_ratio=$3
block_size=$4
sup=$5
fastcoco=$6
masked=$7
kd=$8
batch_size=${9}
epochs=${10}
gpu=${11}
target=${12} # DDR EYEPACS 


target="APTOS"
CUDA_VISIBLE_DEVICES=$gpu python main_v0_test.py --root ./GDRBench/ \
                --algorithm GDRNet_MASK_SIAM \
                --dg_mode DG \
                --source-domains DEEPDR FGADR IDRID MESSIDOR RLDR \
                --target-domains  ${target}  \
                --output ./best_DG/${model}_${mask_ratio}_${block_size}_${lr}_${sup}_${fastcoco}_${masked}_${kd}_${batch_size}_${epochs}/APTOS \
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

target="DEEPDR"
CUDA_VISIBLE_DEVICES=$gpu python main_v0_test.py --root ./GDRBench/ \
                --algorithm GDRNet_MASK_SIAM \
                --dg_mode DG \
                --source-domains APTOS FGADR IDRID MESSIDOR RLDR  \
                --target-domains  ${target} \
                --output ./best_DG/${model}_${mask_ratio}_${block_size}_${lr}_${sup}_${fastcoco}_${masked}_${kd}_${batch_size}_${epochs}/DEEPDR \
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

target="FGADR"
CUDA_VISIBLE_DEVICES=$gpu python main_v0_test.py --root ./GDRBench/ \
                --algorithm GDRNet_MASK_SIAM \
                --dg_mode DG \
                --source-domains APTOS DEEPDR IDRID MESSIDOR RLDR \
                --target-domains  ${target} \
                --output ./best_DG/${model}_${mask_ratio}_${block_size}_${lr}_${sup}_${fastcoco}_${masked}_${kd}_${batch_size}_${epochs}/FGADR \
                --learning_rate $lr \
                --backbone ${model} \
                --mask_ratio $mask_ratio \
                --block_size $block_size \
                --fastmoco $fastcoco \
                --masked $masked \
                --batch_size $batch_size \
                --epochs $epochs \
                --sup $sup \
                --kd $kd 

target="IDRID"
CUDA_VISIBLE_DEVICES=$gpu python main_v0_test.py --root ./GDRBench/ \
                --algorithm GDRNet_MASK_SIAM \
                --dg_mode DG \
                --source-domains APTOS DEEPDR FGADR MESSIDOR RLDR \
                --target-domains  ${target} \
                --output ./best_DG/${model}_${mask_ratio}_${block_size}_${lr}_${sup}_${fastcoco}_${masked}_${kd}_${batch_size}_${epochs}/IDRID \
                --learning_rate $lr \
                --backbone ${model} \
                --mask_ratio $mask_ratio \
                --block_size $block_size \
                --fastmoco $fastcoco \
                --masked $masked \
                --batch_size $batch_size \
                --epochs $epochs \
                --sup $sup \
                --kd $kd 

target="MESSIDOR"
CUDA_VISIBLE_DEVICES=$gpu python main_v0_test.py --root ./GDRBench/ \
                --algorithm GDRNet_MASK_SIAM \
                --dg_mode DG \
                --source-domains APTOS DEEPDR FGADR IDRID RLDR \
                --target-domains  ${target} \
                --output ./best_DG/${model}_${mask_ratio}_${block_size}_${lr}_${sup}_${fastcoco}_${masked}_${kd}_${batch_size}_${epochs}/MESSIDOR \
                --learning_rate $lr \
                --backbone ${model} \
                --mask_ratio $mask_ratio \
                --block_size $block_size \
                --fastmoco $fastcoco \
                --masked $masked \
                --batch_size $batch_size \
                --epochs $epochs \
                --kd $kd 

target="RLDR"
CUDA_VISIBLE_DEVICES=$gpu python main_v0_test.py --root ./GDRBench/ \
                --algorithm GDRNet_MASK_SIAM \
                --dg_mode DG \
                --source-domains APTOS DEEPDR FGADR IDRID MESSIDOR \
                --target-domains  ${target} \
                --output ./best_DG/${model}_${mask_ratio}_${block_size}_${lr}_${sup}_${fastcoco}_${masked}_${kd}_${batch_size}_${epochs}/RLDR \
                --learning_rate $lr \
                --backbone ${model} \
                --mask_ratio $mask_ratio \
                --block_size $block_size \
                --fastmoco $fastcoco \
                --masked $masked \
                --batch_size $batch_size \
                --epochs $epochs \
                --sup $sup \
                --kd $kd 
