
# method=$1

# for source in  APTOS DEEPDR FGADR IDRID MESSIDOR RLDR

# do 
#     python main_v0.py --root ./GDRBench/ \
#                 --algorithm GDRNet_MASK_SIAM \
#                 --dg_mode ESDG \
#                 --source-domains ${source} \
#                 --target-domains DDR EYEPACS \
#                 --output ./GDRNet_Mask_SIAM_FASTMOCO_V0/${source}
# done
                # --target-domains DEEPDR FGADR IDRID MESSIDOR RLDR \

# for source in  APTOS DEEPDR FGADR IDRID MESSIDOR RLDR


# CUDA_VISIBLE_DEVICES=2 python main_v0.py --root ./GDRBench/ --algorithm GDRNet_MASK_SIAM_SIAM  --dg_mode ESDG  --source-domain RLDR --target-domains APTOS DEEPDR FGADR IDRID MESSIDOR DDR EYEPACS  --output ./GDRNet_Mask_SIAM_FASTMOCO_V0/QT_Dual/ESDG/resnet50_resnet50_0.5_32_0.001/RLDR   --backbone resnet50  --mask_ratio 0.5  --block_size 32 --learning_rate 0.001


# CUDA_VISIBLE_DEVICES=3 python main_v0.py --root ./GDRBench/ --algorithm GDRNet_MASK_SIAM  --dg_mode ESDG  --source-domain RLDR --target-domains APTOS DEEPDR FGADR IDRID MESSIDOR DDR EYEPACS  --output ./GDRNet_Mask_SIAM_FASTMOCO_V0/QT_Dual/ESDG/resnet50_resnet50_0.5_32_0.001/RLDR   --backbone resnet50  --mask_ratio 0.5  --block_size 32 --learning_rate 0.001 --fastmoco $fastcoco




# bash main_ESDG_mask_siam_fastmoco.sh resnet50 0.001 0.5 32 0.0 16 4

# bash main_ESDG_mask_siam_fastmoco.sh resnet50 0.00045 0.6 32 0.6 1.5 0.6 0.1 0.0 0.4



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
smooth=${11}
batch_size=${12}
epochs=${13}
gpu=${14}



CUDA_VISIBLE_DEVICES=$gpu python main_v0.py --root ./GDRBench/ \
                --algorithm GDRNet_MASK_SIAM \
                --dg_mode ESDG \
                --source-domains APTOS \
                --target-domains DEEPDR FGADR IDRID MESSIDOR RLDR DDR EYEPACS \
                --output ./GDRNet_Mask_SIAM_FASTMOCO_V0/${model}_${mask_ratio}_${block_size}_${lr}_${sup}_${fastcoco}_${masked}_${kd}_${k}_${p}_${smooth}_${batch_size}_${epochs}/APTOS \
                --learning_rate $lr \
                --backbone ${model} \
                --mask_ratio $3 \
                --block_size $4 \
                --fastmoco $fastcoco \
                --masked $masked \
                --batch_size $batch_size \
                --epochs $epochs \
                --sup $sup \
                --kd $kd \
                --positive $k \
                --p $p \
                --smooth $smooth


CUDA_VISIBLE_DEVICES=$gpu python main_v0.py --root ./GDRBench/ \
                --algorithm GDRNet_MASK_SIAM \
                --dg_mode ESDG \
                --source-domains DEEPDR \
                --target-domains APTOS FGADR IDRID MESSIDOR RLDR DDR EYEPACS \
                --output ./GDRNet_Mask_SIAM_FASTMOCO_V0/${model}_${mask_ratio}_${block_size}_${lr}_${sup}_${fastcoco}_${masked}_${kd}_${k}_${p}_${smooth}_${batch_size}_${epochs}/DEEPDR \
                --learning_rate $lr \
                --backbone ${model} \
                --mask_ratio $3 \
                --block_size $4 \
                --fastmoco $fastcoco \
                --masked $masked \
                --batch_size $batch_size \
                --epochs $epochs \
                --sup $sup \
                --kd $kd \
                --positive $k \
                --p $p \
                --smooth $smooth



CUDA_VISIBLE_DEVICES=$gpu python main_v0.py --root ./GDRBench/ \
                --algorithm GDRNet_MASK_SIAM \
                --dg_mode ESDG \
                --source-domains FGADR \
                --target-domains APTOS DEEPDR IDRID MESSIDOR RLDR DDR EYEPACS \
                --output ./GDRNet_Mask_SIAM_FASTMOCO_V0/${model}_${mask_ratio}_${block_size}_${lr}_${sup}_${fastcoco}_${masked}_${kd}_${k}_${p}_${smooth}_${batch_size}_${epochs}/FGADR \
                --learning_rate $lr \
                --backbone ${model} \
                --mask_ratio $mask_ratio \
                --block_size $block_size \
                --fastmoco $fastcoco \
                --masked $masked \
                --batch_size $batch_size \
                --epochs $epochs \
                --sup $sup \
                --kd $kd \
                --positive $k \
                --p $p \
                --smooth $smooth

CUDA_VISIBLE_DEVICES=$gpu python main_v0.py --root ./GDRBench/ \
                --algorithm GDRNet_MASK_SIAM \
                --dg_mode ESDG \
                --source-domains IDRID \
                --target-domains APTOS DEEPDR FGADR MESSIDOR RLDR DDR EYEPACS \
                --output ./GDRNet_Mask_SIAM_FASTMOCO_V0/${model}_${mask_ratio}_${block_size}_${lr}_${sup}_${fastcoco}_${masked}_${kd}_${k}_${p}_${smooth}_${batch_size}_${epochs}/IDRID \
                --learning_rate $lr \
                --backbone ${model} \
                --mask_ratio $mask_ratio \
                --block_size $block_size \
                --fastmoco $fastcoco \
                --masked $masked \
                --batch_size $batch_size \
                --epochs $epochs \
                --sup $sup \
                --kd $kd \
                --positive $k \
                --p $p \
                --smooth $smooth


CUDA_VISIBLE_DEVICES=$gpu python main_v0.py --root ./GDRBench/ \
                --algorithm GDRNet_MASK_SIAM \
                --dg_mode ESDG \
                --source-domains MESSIDOR \
                --target-domains APTOS DEEPDR FGADR IDRID RLDR DDR EYEPACS \
                --output ./GDRNet_Mask_SIAM_FASTMOCO_V0/${model}_${mask_ratio}_${block_size}_${lr}_${sup}_${fastcoco}_${masked}_${kd}_${k}_${p}_${smooth}_${batch_size}_${epochs}/MESSIDOR \
                --learning_rate $lr \
                --backbone ${model} \
                --mask_ratio $mask_ratio \
                --block_size $block_size \
                --fastmoco $fastcoco \
                --masked $masked \
                --batch_size $batch_size \
                --epochs $epochs \
                --kd $kd \
                --positive $k \
                --p $p \
                --smooth $smooth


CUDA_VISIBLE_DEVICES=$gpu python main_v0.py --root ./GDRBench/ \
                --algorithm GDRNet_MASK_SIAM \
                --dg_mode ESDG \
                --source-domains RLDR \
                --target-domains APTOS DEEPDR FGADR IDRID MESSIDOR DDR EYEPACS \
                --output ./GDRNet_Mask_SIAM_FASTMOCO_V0/${model}_${mask_ratio}_${block_size}_${lr}_${sup}_${fastcoco}_${masked}_${kd}_${k}_${p}_${smooth}_${batch_size}_${epochs}/RLDR \
                --learning_rate $lr \
                --backbone ${model} \
                --mask_ratio $mask_ratio \
                --block_size $block_size \
                --fastmoco $fastcoco \
                --masked $masked \
                --batch_size $batch_size \
                --epochs $epochs \
                --sup $sup \
                --kd $kd \
                --positive $k \
                --p $p \
                --smooth $smooth

