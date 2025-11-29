
# method=$1

# for source in  APTOS DEEPDR FGADR IDRID MESSIDOR RLDR

# do 
#     python main_qt_dual.py --root ./GDRBench/ \
#                 --algorithm GDRNet_DUAL_MASK \
#                 --dg_mode ESDG \
#                 --source-domain ${source} \
#                 --target-domains DDR EYEPACS \
#                 --output ./GDRNet_DUAL_MASK/${source}
# done
                # --target-domains DEEPDR FGADR IDRID MESSIDOR RLDR \

# for source in  APTOS DEEPDR FGADR IDRID MESSIDOR RLDR

# do 

# python main_qt_dual.py --root ./GDRBench/ \
#                 --algorithm GDRNet_DUAL_MASK \
#                 --dg_mode ESDG \
#                 --source-domain APTOS \
#                 --target-domains DEEPDR FGADR IDRID MESSIDOR RLDR DDR EYEPACS \
#                 --output ./GDRNet_DUAL_MASK/QT/APTOS 
#                 # --backbone ${model}



model1=$1
model2=$2
mask_ratio=$3
block_size=$4
gpu=$5
CUDA_VISIBLE_DEVICES=$5 python main_qt_dual.py --root ./GDRBench/ \
                --algorithm GDRNet_DUAL_MASK \
                --dg_mode ESDG \
                --source-domain APTOS \
                --target-domains DEEPDR FGADR IDRID MESSIDOR RLDR DDR EYEPACS \
                --output ./GDRNet_DUAL_MASK/QT_Dual/ESDG/${model1}_${model2}_${mask_ratio}_${block_size}/APTOS \
                --backbone1 ${model1} \
                --backbone2 ${model2} \
                --mask_ratio $3 \
                --block_size $4


CUDA_VISIBLE_DEVICES=$5 python main_qt_dual.py --root ./GDRBench/ \
                --algorithm GDRNet_DUAL_MASK \
                --dg_mode ESDG \
                --source-domain DEEPDR \
                --target-domains APTOS FGADR IDRID MESSIDOR RLDR DDR EYEPACS \
                --output ./GDRNet_DUAL_MASK/QT_Dual/ESDG/${model1}_${model2}_${mask_ratio}_${block_size}/DEEPDR \
                --backbone1 ${model1} \
                --backbone2 ${model2} \
                --mask_ratio $3 \
                --block_size $4




CUDA_VISIBLE_DEVICES=$5 python main_qt_dual.py --root ./GDRBench/ \
                --algorithm GDRNet_DUAL_MASK \
                --dg_mode ESDG \
                --source-domain FGADR \
                --target-domains APTOS DEEPDR IDRID MESSIDOR RLDR DDR EYEPACS \
                --output ./GDRNet_DUAL_MASK/QT_Dual/ESDG/${model1}_${model2}_${mask_ratio}_${block_size}/FGADR \
                --backbone1 ${model1} \
                --backbone2 ${model2} \
                --mask_ratio $3 \
                --block_size $4

CUDA_VISIBLE_DEVICES=$5 python main_qt_dual.py --root ./GDRBench/ \
                --algorithm GDRNet_DUAL_MASK \
                --dg_mode ESDG \
                --source-domain IDRID \
                --target-domains APTOS DEEPDR FGADR MESSIDOR RLDR DDR EYEPACS \
                --output ./GDRNet_DUAL_MASK/QT_Dual/ESDG/${model1}_${model2}_${mask_ratio}_${block_size}/IDRID \
                --backbone1 ${model1} \
                --backbone2 ${model2} \
                --mask_ratio $3 \
                --block_size $4

CUDA_VISIBLE_DEVICES=$5 python main_qt_dual.py --root ./GDRBench/ \
                --algorithm GDRNet_DUAL_MASK \
                --dg_mode ESDG \
                --source-domain MESSIDOR \
                --target-domains APTOS DEEPDR FGADR IDRID RLDR DDR EYEPACS \
                --output ./GDRNet_DUAL_MASK/QT_Dual/ESDG/${model1}_${model2}_${mask_ratio}_${block_size}/MESSIDOR \
                --backbone1 ${model1} \
                --backbone2 ${model2} \
                --mask_ratio $3 \
                --block_size $4

CUDA_VISIBLE_DEVICES=$5 python main_qt_dual.py --root ./GDRBench/ \
                --algorithm GDRNet_DUAL_MASK \
                --dg_mode ESDG \
                --source-domain RLDR \
                --target-domains APTOS DEEPDR FGADR IDRID MESSIDOR DDR EYEPACS \
                --output ./GDRNet_DUAL_MASK/QT_Dual/ESDG/${model1}_${model2}_${mask_ratio}_${block_size}/RLDR \
                --backbone1 ${model1} \
                --backbone2 ${model2} \
                --mask_ratio $3 \
                --block_size $4
