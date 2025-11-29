
# method=$1

# for source in  APTOS DEEPDR FGADR IDRID MESSIDOR RLDR

# do 
#     python main.py --root ./GDRBench/ \
#                 --algorithm GDRNet_MASK_SIAM \
#                 --dg_mode ESDG \
#                 --source-domains ${source} \
#                 --target-domains DDR EYEPACS \
#                 --output ./GDRNet_Mask_SIAM_FASTMOCO_MIXUP/${source}
# done
                # --target-domains DEEPDR FGADR IDRID MESSIDOR RLDR \

# for source in  APTOS DEEPDR FGADR IDRID MESSIDOR RLDR


# CUDA_VISIBLE_DEVICES=2 python main.py --root ./GDRBench/ --algorithm GDRNet_MASK_SIAM_SIAM  --dg_mode ESDG  --source-domain RLDR --target-domains APTOS DEEPDR FGADR IDRID MESSIDOR DDR EYEPACS  --output ./GDRNet_Mask_SIAM_FASTMOCO_MIXUP/QT_Dual/ESDG/resnet50_resnet50_0.5_32_0.001/RLDR   --backbone resnet50  --mask_ratio 0.5  --block_size 32 --learning_rate 0.001


# CUDA_VISIBLE_DEVICES=3 python main.py --root ./GDRBench/ --algorithm GDRNet_MASK_SIAM  --dg_mode ESDG  --source-domain RLDR --target-domains APTOS DEEPDR FGADR IDRID MESSIDOR DDR EYEPACS  --output ./GDRNet_Mask_SIAM_FASTMOCO_MIXUP/QT_Dual/ESDG/resnet50_resnet50_0.5_32_0.001/RLDR   --backbone resnet50  --mask_ratio 0.5  --block_size 32 --learning_rate 0.001 --fastmoco True


# do 

model=$1
lr=$2
mask_ratio=$3
block_size=$4
gpu=$5


CUDA_VISIBLE_DEVICES=$5 python main.py --root ./GDRBench/ \
                --algorithm GDRNet_MASK_SIAM \
                --dg_mode ESDG \
                --source-domains APTOS \
                --target-domains DEEPDR FGADR IDRID MESSIDOR RLDR DDR EYEPACS \
                --output ./GDRNet_Mask_SIAM_FASTMOCO_MIXUP/${model}_${mask_ratio}_${block_size}_${lr}/APTOS \
                --learning_rate $lr \
                --backbone ${model} \
                --mask_ratio $3 \
                --block_size $4 \
                --fastmoco True

CUDA_VISIBLE_DEVICES=$5 python main.py --root ./GDRBench/ \
                --algorithm GDRNet_MASK_SIAM \
                --dg_mode ESDG \
                --source-domains DEEPDR \
                --target-domains APTOS FGADR IDRID MESSIDOR RLDR DDR EYEPACS \
                --output ./GDRNet_Mask_SIAM_FASTMOCO_MIXUP/${model}_${mask_ratio}_${block_size}_${lr}/DEEPDR \
                --learning_rate $lr \
                --backbone ${model} \
                --mask_ratio $3 \
                --block_size $4 \
                --fastmoco True

CUDA_VISIBLE_DEVICES=$5 python main.py --root ./GDRBench/ \
                --algorithm GDRNet_MASK_SIAM \
                --dg_mode ESDG \
                --source-domains FGADR \
                --target-domains APTOS DEEPDR IDRID MESSIDOR RLDR DDR EYEPACS \
                --output ./GDRNet_Mask_SIAM_FASTMOCO_MIXUP/${model}_${mask_ratio}_${block_size}_${lr}/FGADR \
                --learning_rate $lr \
                --backbone ${model} \
                --mask_ratio $3 \
                --block_size $4 \
                --fastmoco True

CUDA_VISIBLE_DEVICES=$5 python main.py --root ./GDRBench/ \
                --algorithm GDRNet_MASK_SIAM \
                --dg_mode ESDG \
                --source-domains IDRID \
                --target-domains APTOS DEEPDR FGADR MESSIDOR RLDR DDR EYEPACS \
                --output ./GDRNet_Mask_SIAM_FASTMOCO_MIXUP/${model}_${mask_ratio}_${block_size}_${lr}/IDRID \
                --learning_rate $lr \
                --backbone ${model} \
                --mask_ratio $3 \
                --block_size $4 \
                --fastmoco True

CUDA_VISIBLE_DEVICES=$5 python main.py --root ./GDRBench/ \
                --algorithm GDRNet_MASK_SIAM \
                --dg_mode ESDG \
                --source-domains MESSIDOR \
                --target-domains APTOS DEEPDR FGADR IDRID RLDR DDR EYEPACS \
                --output ./GDRNet_Mask_SIAM_FASTMOCO_MIXUP/${model}_${mask_ratio}_${block_size}_${lr}/MESSIDOR \
                --learning_rate $lr \
                --backbone ${model} \
                --mask_ratio $3 \
                --block_size $4 \
                --fastmoco True

CUDA_VISIBLE_DEVICES=$5 python main.py --root ./GDRBench/ \
                --algorithm GDRNet_MASK_SIAM \
                --dg_mode ESDG \
                --source-domains RLDR \
                --target-domains APTOS DEEPDR FGADR IDRID MESSIDOR DDR EYEPACS \
                --output ./GDRNet_Mask_SIAM_FASTMOCO_MIXUP/${model}_${mask_ratio}_${block_size}_${lr}/RLDR \
                --learning_rate $lr \
                --backbone ${model} \
                --mask_ratio $3 \
                --block_size $4 \
                --fastmoco True