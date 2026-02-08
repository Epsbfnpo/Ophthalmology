
# method=$1

# for source in  APTOS DEEPDR FGADR IDRID MESSIDOR RLDR

# do 
#     python main_qt.py --root ./GDRBench/ \
#                 --algorithm GDRNet \
#                 --dg_mode ESDG \
#                 --source-domain ${source} \
#                 --target-domains DDR EYEPACS \
#                 --output ./GDRNet/${source}
# done
                # --target-domains DEEPDR FGADR IDRID MESSIDOR RLDR \

# for source in  APTOS DEEPDR FGADR IDRID MESSIDOR RLDR

# do 

# python main_qt.py --root ./GDRBench/ \
#                 --algorithm GDRNet \
#                 --dg_mode ESDG \
#                 --source-domain APTOS \
#                 --target-domains DEEPDR FGADR IDRID MESSIDOR RLDR DDR EYEPACS \
#                 --output ./GDRNet/QT/APTOS 
#                 # --backbone ${model}



model=$1
python main_qt.py --root ./GDRBench/ \
                --algorithm GDRNet \
                --dg_mode ESDG \
                --source-domain APTOS \
                --target-domains DEEPDR FGADR IDRID MESSIDOR RLDR DDR EYEPACS \
                --output ./GDRNet/QT/ESDG/${model}/APTOS \
                --backbone ${model}

python main_qt.py --root ./GDRBench/ \
                --algorithm GDRNet \
                --dg_mode ESDG \
                --source-domain DEEPDR \
                --target-domains APTOS FGADR IDRID MESSIDOR RLDR DDR EYEPACS \
                --output ./GDRNet/QT/ESDG/${model}/DEEPDR \
                --backbone ${model}


python main_qt.py --root ./GDRBench/ \
                --algorithm GDRNet \
                --dg_mode ESDG \
                --source-domain FGADR \
                --target-domains APTOS DEEPDR IDRID MESSIDOR RLDR DDR EYEPACS \
                --output ./GDRNet/QT/ESDG/${model}/FGADR \
                --backbone ${model}

python main_qt.py --root ./GDRBench/ \
                --algorithm GDRNet \
                --dg_mode ESDG \
                --source-domain IDRID \
                --target-domains APTOS DEEPDR FGADR MESSIDOR RLDR DDR EYEPACS \
                --output ./GDRNet/QT/ESDG/${model}/IDRID \
                --backbone ${model}

python main_qt.py --root ./GDRBench/ \
                --algorithm GDRNet \
                --dg_mode ESDG \
                --source-domain MESSIDOR \
                --target-domains APTOS DEEPDR FGADR IDRID RLDR DDR EYEPACS \
                --output ./GDRNet/QT/ESDG/${model}/MESSIDOR \
                --backbone ${model}


python main_qt.py --root ./GDRBench/ \
                --algorithm GDRNet \
                --dg_mode ESDG \
                --source-domain RLDR \
                --target-domains APTOS DEEPDR FGADR IDRID MESSIDOR DDR EYEPACS \
                --output ./GDRNet/QT/ESDG/${model}/RLDR \
                --backbone ${model}
                