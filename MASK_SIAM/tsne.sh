model_path="/home/user/Projects/Medical-Image/Ophthalmology/DGDR/result/fundusaug/DG/resnet50_0.5_32_0.00045_0.6_1.5_0.6_0.05_32_100/APTOS"

python eval_tsne.py  --root ./GDRBench/ \
                --algorithm GDRNet_MASK_SIAM \
                --dg_mode DG \
                --source-domains DEEPDR FGADR IDRID MESSIDOR RLDR \
                --target-domains APTOS  \
                --model_path /home/user/Projects/Medical-Image/Ophthalmology/DGDR/result/fundusaug/DG/resnet50_0.5_32_0.00045_0.6_1.5_0.6_0.05_32_100/APTOS


python eval_tsne.py  --root ./GDRBench/ \
                --algorithm GDRNet_MASK_SIAM \
                --dg_mode DG \
                --source-domains APTOS FGADR IDRID MESSIDOR RLDR \
                --target-domains DEEPDR  \
                --model_path /home/user/Projects/Medical-Image/Ophthalmology/DGDR/result/fundusaug/DG/resnet50_0.5_32_0.00045_0.6_1.5_0.6_0.05_32_100/DEEPDR
            

python eval_tsne.py  --root ./GDRBench/ \
                --algorithm GDRNet_MASK_SIAM \
                --dg_mode DG \
                --source-domains APTOS DEEPDR IDRID MESSIDOR RLDR \
                --target-domains FGADR  \
                --model_path /home/user/Projects/Medical-Image/Ophthalmology/DGDR/result/fundusaug/DG/resnet50_0.5_32_0.00045_0.6_1.5_0.6_0.05_32_100/FGADR

python eval_tsne.py  --root ./GDRBench/ \
                --algorithm GDRNet_MASK_SIAM \
                --dg_mode DG \
                --source-domains APTOS DEEPDR FGADR  MESSIDOR RLDR \
                --target-domains IDRID  \
                --model_path /home/user/Projects/Medical-Image/Ophthalmology/DGDR/result/fundusaug/DG/resnet50_0.5_32_0.00045_0.6_1.5_0.6_0.05_32_100/IDRID

python eval_tsne.py  --root ./GDRBench/ \
                --algorithm GDRNet_MASK_SIAM \
                --dg_mode DG \
                --source-domains APTOS DEEPDR FGADR IDRID RLDR \
                --target-domains MESSIDOR  \
                --model_path /home/user/Projects/Medical-Image/Ophthalmology/DGDR/result/fundusaug/DG/resnet50_0.5_32_0.00045_0.6_1.5_0.6_0.05_32_100/MESSIDOR 

python eval_tsne.py  --root ./GDRBench/ \
                --algorithm GDRNet_MASK_SIAM \
                --dg_mode DG \
                --source-domains APTOS DEEPDR FGADR IDRID MESSIDOR \
                --target-domains RLDR  \
                --model_path /home/user/Projects/Medical-Image/Ophthalmology/DGDR/result/fundusaug/DG/resnet50_0.5_32_0.00045_0.6_1.5_0.6_0.05_32_100/RLDR 