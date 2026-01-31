
# method=$1

# for source in  APTOS DEEPDR FGADR IDRID MESSIDOR RLDR

# do 
#     python main.py --root ./GDRBench/ \
#                 --algorithm GDRNet \
#                 --dg_mode ESDG \
#                 --source-domains ${source} \
#                 --target-domains DDR EYEPACS \
#                 --output ./GDRNet/${source}
# done
                # --target-domains DEEPDR FGADR IDRID MESSIDOR RLDR \

# for source in  APTOS DEEPDR FGADR IDRID MESSIDOR RLDR

# do 
model=$1
gpu=$2
lr=$3
CUDA_VISIBLE_DEVICES=$2 python main.py --root ./GDRBench/ \
                --algorithm GDRNet \
                --dg_mode ESDG \
                --source-domains APTOS \
                --target-domains DEEPDR FGADR IDRID MESSIDOR RLDR DDR EYEPACS \
                --output ./GDRNet/${model}_${lr}/APTOS \
                --learning_rate $3 \
                --backbone ${model}

CUDA_VISIBLE_DEVICES=$2 python main.py --root ./GDRBench/ \
                --algorithm GDRNet \
                --dg_mode ESDG \
                --source-domains DEEPDR \
                --target-domains APTOS FGADR IDRID MESSIDOR RLDR DDR EYEPACS \
                --output ./GDRNet/${model}_${lr}/DEEPDR \
                --learning_rate $3 \
                --backbone ${model}


CUDA_VISIBLE_DEVICES=$2 python main.py --root ./GDRBench/ \
                --algorithm GDRNet \
                --dg_mode ESDG \
                --source-domains FGADR \
                --target-domains APTOS DEEPDR IDRID MESSIDOR RLDR DDR EYEPACS \
                --output ./GDRNet/${model}_${lr}/FGADR \
                --learning_rate $3 \
                --backbone ${model}

CUDA_VISIBLE_DEVICES=$2 python main.py --root ./GDRBench/ \
                --algorithm GDRNet \
                
                --dg_mode ESDG \
                --source-domains IDRID \
                --target-domains APTOS DEEPDR FGADR MESSIDOR RLDR DDR EYEPACS \
                --output ./GDRNet/${model}_${lr}/IDRID \
                --learning_rate $3 \
                --backbone ${model}

CUDA_VISIBLE_DEVICES=$2 python main.py --root ./GDRBench/ \
                --algorithm GDRNet \
                --dg_mode ESDG \
                --source-domains MESSIDOR \
                --target-domains APTOS DEEPDR FGADR IDRID RLDR DDR EYEPACS \
                --output ./GDRNet/${model}_${lr}/MESSIDOR \
                --learning_rate $3 \
                --backbone ${model}


CUDA_VISIBLE_DEVICES=$2 python main.py --root ./GDRBench/ \
                --algorithm GDRNet \
                --dg_mode ESDG \
                --source-domains RLDR \
                --target-domains APTOS DEEPDR FGADR IDRID MESSIDOR DDR EYEPACS \
                --output ./GDRNet/${model}_${lr}/RLDR \
                --learning_rate $3 \
                --backbone ${model}
                