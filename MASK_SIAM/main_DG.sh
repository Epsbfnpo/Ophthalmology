
# APTOS DEEPDR FGADR IDRID MESSIDOR RLDR


# python main.py --root YOUR_DATASET_ROOT_PATH
#                --algorithm DRGen
#                --dg_mode DG
#                --source-domains DATASET2 DATASET3 ...
#                --target-domains DATASET1 
#                --output YOUR_OUTPUT_DIR

python main.py --root /ai/Lee/Projects/Medical-Image/Ophthalmology/DGDR/GDRBench/ \
                --algorithm GDRNet \
                --dg_mode DG \
                --source-domains DEEPDR FGADR IDRID MESSIDOR RLDR \
                --target-domains APTOS  \
                --output ./GDRNet/DG/APTOS

python main.py --root /ai/Lee/Projects/Medical-Image/Ophthalmology/DGDR/GDRBench/ \
                --algorithm GDRNet \
                --dg_mode DG \
                --source-domains APTOS FGADR IDRID MESSIDOR RLDR \
                --target-domains DEEPDR  \
                --output ./GDRNet/DG/DEEPDR

python main.py --root /ai/Lee/Projects/Medical-Image/Ophthalmology/DGDR/GDRBench/ \
                --algorithm GDRNet \
                --dg_mode DG \
                --source-domains APTOS DEEPDR IDRID MESSIDOR RLDR \
                --target-domains FGADR  \
                --output ./GDRNet/DG/FGADR

python main.py --root /ai/Lee/Projects/Medical-Image/Ophthalmology/DGDR/GDRBench/ \
                --algorithm GDRNet \
                --dg_mode DG \
                --source-domains APTOS DEEPDR FGADR MESSIDOR RLDR \
                --target-domains IDRID  \
                --output ./GDRNet/DG/IDRID

python main.py --root /ai/Lee/Projects/Medical-Image/Ophthalmology/DGDR/GDRBench/ \
                --algorithm GDRNet \
                --dg_mode DG \
                --source-domains APTOS DEEPDR FGADR IDRID RLDR \
                --target-domains MESSIDOR  \
                --output ./GDRNet/DG/MESSIDOR

python main.py --root /ai/Lee/Projects/Medical-Image/Ophthalmology/DGDR/GDRBench/ \
                --algorithm GDRNet \
                --dg_mode DG \
                --source-domains APTOS DEEPDR FGADR IDRID MESSIDOR \
                --target-domains RLDR  \
                --output ./GDRNet/DG/RLDR