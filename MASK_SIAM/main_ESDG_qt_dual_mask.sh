
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
gpu=$3


CUDA_VISIBLE_DEVICES=$3 python main_qt_dual.py --root ./GDRBench/ \
                --algorithm GDRNet_DUAL_MASK \
                --dg_mode ESDG \
                --source-domain APTOS \
                --target-domains DEEPDR FGADR IDRID MESSIDOR RLDR DDR EYEPACS \
                --output ./GDRNet_DUAL_MASK/QT_Dual/ESDG/${model1}_${model2}/APTOS \
                --backbone1 ${model1} \
                --backbone2 ${model2}


CUDA_VISIBLE_DEVICES=$3 python main_qt_dual.py --root ./GDRBench/ \
                --algorithm GDRNet_DUAL_MASK \
                --dg_mode ESDG \
                --source-domain DEEPDR \
                --target-domains APTOS FGADR IDRID MESSIDOR RLDR DDR EYEPACS \
                --output ./GDRNet_DUAL_MASK/QT_Dual/ESDG/${model1}_${model2}/DEEPDR \
                --backbone1 ${model1} \
                --backbone2 ${model2}

CUDA_VISIBLE_DEVICES=$3 python main_qt_dual.py --root ./GDRBench/ \
                --algorithm GDRNet_DUAL_MASK \
                --dg_mode ESDG \
                --source-domain FGADR \
                --target-domains APTOS DEEPDR IDRID MESSIDOR RLDR DDR EYEPACS \
                --output ./GDRNet_DUAL_MASK/QT_Dual/ESDG/${model1}_${model2}/FGADR \
                --backbone1 ${model1} \
                --backbone2 ${model2}

CUDA_VISIBLE_DEVICES=$3 python main_qt_dual.py --root ./GDRBench/ \
                --algorithm GDRNet_DUAL_MASK \
                --dg_mode ESDG \
                --source-domain IDRID \
                --target-domains APTOS DEEPDR FGADR MESSIDOR RLDR DDR EYEPACS \
                --output ./GDRNet_DUAL_MASK/QT_Dual/ESDG/${model1}_${model2}/IDRID \
                --backbone1 ${model1} \
                --backbone2 ${model2}

CUDA_VISIBLE_DEVICES=$3 python main_qt_dual.py --root ./GDRBench/ \
                --algorithm GDRNet_DUAL_MASK \
                --dg_mode ESDG \
                --source-domain MESSIDOR \
                --target-domains APTOS DEEPDR FGADR IDRID RLDR DDR EYEPACS \
                --output ./GDRNet_DUAL_MASK/QT_Dual/ESDG/${model1}_${model2}/MESSIDOR \
                --backbone1 ${model1} \
                --backbone2 ${model2}

CUDA_VISIBLE_DEVICES=$3 python main_qt_dual.py --root ./GDRBench/ \
                --algorithm GDRNet_DUAL_MASK \
                --dg_mode ESDG \
                --source-domain RLDR \
                --target-domains APTOS DEEPDR FGADR IDRID MESSIDOR DDR EYEPACS \
                --output ./GDRNet_DUAL_MASK/QT_Dual/ESDG/${model1}_${model2}/RLDR \
                --backbone1 ${model1} \
                --backbone2 ${model2}                