
# method=$1

# for source in  APTOS DEEPDR FGADR IDRID MESSIDOR RLDR

# do 
#     python main_qt.py --root /ai/Lee/Projects/Medical-Image/Ophthalmology/DGDR/GDRBench/ \
#                 --algorithm GDRNet \
#                 --dg_mode ESDG \
#                 --source-domain ${source} \
#                 --target-domains DDR EYEPACS \
#                 --output ./GDRNet/${source}
# done
                # --target-domains DEEPDR FGADR IDRID MESSIDOR RLDR \

# for source in  APTOS DEEPDR FGADR IDRID MESSIDOR RLDR

# do 

# python main_qt.py --root /ai/Lee/Projects/Medical-Image/Ophthalmology/DGDR/GDRBench/ \
#                 --algorithm GDRNet \
#                 --dg_mode ESDG \
#                 --source-domain APTOS \
#                 --target-domains DEEPDR FGADR IDRID MESSIDOR RLDR DDR EYEPACS \
#                 --output ./GDRNet/QT/APTOS 
#                 # --backbone ${model}



model=$1
python main_qt.py --root /ai/Lee/Projects/Medical-Image/Ophthalmology/DGDR/GDRBench/ \
                --algorithm GDRNet \
                --dg_mode DG \
                --source-domain DEEPDR FGADR IDRID MESSIDOR RLDR \
                --target-domains  APTOS \
                --output ./GDRNet/QT/DG/${model}/APTOS \
                --backbone ${model}

python main_qt.py --root /ai/Lee/Projects/Medical-Image/Ophthalmology/DGDR/GDRBench/ \
                --algorithm GDRNet \
                --dg_mode DG \
                --source-domain APTOS FGADR IDRID MESSIDOR RLDR \
                --target-domains DEEPDR  \
                --output ./GDRNet/QT/DG/${model}/DEEPDR \
                --backbone ${model}


python main_qt.py --root /ai/Lee/Projects/Medical-Image/Ophthalmology/DGDR/GDRBench/ \
                --algorithm GDRNet \
                --dg_mode DG \
                --source-domain APTOS DEEPDR IDRID MESSIDOR RLDR  \
                --target-domains FGADR\
                --output ./GDRNet/QT/DG/${model}/FGADR \
                --backbone ${model}

python main_qt.py --root /ai/Lee/Projects/Medical-Image/Ophthalmology/DGDR/GDRBench/ \
                --algorithm GDRNet \
                --dg_mode DG \
                --source-domain APTOS DEEPDR FGADR MESSIDOR RLDR \
                --target-domains IDRID  \
                --output ./GDRNet/QT/DG/${model}/IDRID \
                --backbone ${model}

python main_qt.py --root /ai/Lee/Projects/Medical-Image/Ophthalmology/DGDR/GDRBench/ \
                --algorithm GDRNet \
                --dg_mode DG \
                --source-domain APTOS DEEPDR FGADR IDRID RLDR \
                --target-domains MESSIDOR \
                --output ./GDRNet/QT/DG/${model}/MESSIDOR \
                --backbone ${model}


python main_qt.py --root /ai/Lee/Projects/Medical-Image/Ophthalmology/DGDR/GDRBench/ \
                --algorithm GDRNet \
                --dg_mode DG \
                --source-domain APTOS DEEPDR FGADR IDRID MESSIDOR  \
                --target-domains RLDR  \
                --output ./GDRNet/QT/DG/${model}/RLDR \
                --backbone ${model}
                