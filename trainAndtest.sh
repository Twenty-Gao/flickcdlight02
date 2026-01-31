#export CUBLAS_WORKSPACE_CONFIG=:4096:8
#mode option: train or test
#python train.py --title 'FlickCD_SYSU'\
#                --data_name 'SYSU'\
#                --mode 'train'\
#                --train_dataset_path  '../Dataset/SYSU/train/'\
#                --train_list_path '../Dataset/SYSU/train.txt'\
#                --val_dataset_path '../Dataset/SYSU/val/'\
#                --val_list_path '../Dataset/SYSU/val.txt'\
#                --test_dataset_path '../Dataset/SYSU/test/'\
#                --test_list_path '../Dataset/SYSU/test.txt'\
#                --learning_rate 5e-4\
#                --epochs 100\

#python train.py --title 'FlickCD_WHU'\
#                --data_name 'WHU'\
#                --mode 'train'\
#                --train_dataset_path  '../Dataset/WHU-CD/train/'\
#                --train_list_path '../Dataset/WHU-CD/train.txt'\
#                --val_dataset_path '../Dataset/WHU-CD/val/'\
#                --val_list_path '../Dataset/WHU-CD/val.txt'\
#                --test_dataset_path '../Dataset/WHU-CD/test/'\
#                --test_list_path '../Dataset/WHU-CD/test.txt'\
#                --learning_rate 2e-4\
#                --epochs 100\

#python train.py --title 'FlickCD_LEVIR-CD+'\
#                --data_name 'LEVIR+'\
#                --mode 'train'\
#                --train_dataset_path  '../Dataset/LEVIR-CD+/train/'\
#                --train_list_path '../Dataset/LEVIR-CD+/train.txt'\
#                --val_dataset_path '../Dataset/LEVIR-CD+/val/'\
#                --val_list_path '../Dataset/LEVIR-CD+/val.txt'\
#                --test_dataset_path '../Dataset/LEVIR-CD+/test/'\
#                --test_list_path '../Dataset/LEVIR-CD+/test.txt'\
#                --learning_rate 5e-4\
#                --epochs 200\

#python train.py --title 'FlickCD_CDD'\
#                --data_name 'CDD'\
#                --mode 'train'\
#                --train_dataset_path  '../Dataset/CDD/train/'\
#                --train_list_path '../Dataset/CDD/train.txt'\
#                --val_dataset_path '../Dataset/CDD/val/'\
#                --val_list_path '../Dataset/CDD/val.txt'\
#                --test_dataset_path '../Dataset/CDD/test/'\
#                --test_list_path '../Dataset/CDD/test.txt'\
#                --learning_rate 5e-4\
#                --epochs 250\

python train.py \
	--mode test \
	 --title "LEVIR_CD_Experiment26"\
    --data_name "LEVIR+" \
    --train_dataset_path "../../Dataset/LEVIR-CD+/train/" \
    --train_list_path "../../Dataset/LEVIR-CD+/train.txt" \
    --val_dataset_path "../../Dataset/LEVIR-CD+/val/" \
    --val_list_path "../../Dataset/LEVIR-CD+/val.txt" \
    --test_dataset_path "../../Dataset/LEVIR-CD+/test/" \
    --test_list_path "../../Dataset/LEVIR-CD+/test.txt" \
    --savedir "./result/"

  python train.py \
    --mode test \
    --title "LEVIR_CD_Experiment31" \
    --data_name "SYSU" \
    --train_dataset_path "./dataset/SYSU/SYSU/train/" \
    --train_list_path "./dataset/SYSU/SYSU/train.txt" \
    --val_dataset_path "./dataset/SYSU/SYSU/val/" \
    --val_list_path "./dataset/SYSU/SYSU/val.txt" \
    --test_dataset_path "./dataset/SYSU/SYSU/test/" \
    --test_list_path "./dataset/SYSU/SYSU/test.txt" \
    --savedir "./result/"

  python train_distillation04.py \
    --mode train \
    --title "LEVIR_CD_Experiment_distillation50" \
    --data_name "LEVIR+" \
    --train_dataset_path "../../Dataset/LEVIR-CD+/train/" \
    --train_list_path "../../Dataset/LEVIR-CD+/train.txt" \
    --val_dataset_path "../../Dataset/LEVIR-CD+/val/" \
    --val_list_path "../../Dataset/LEVIR-CD+/val.txt" \
    --test_dataset_path "../../Dataset/LEVIR-CD+/test/" \
    --test_list_path "../../Dataset/LEVIR-CD+/test.txt" \
    --savedir "./result/"\
    --learning_rate 5e-4\
    --epochs 500\
    --resume "./result/LEVIR_CD_Experiment_distillation49/checkpoint_epoch_150.pth"

    python train_distillation02.py \
    --mode train \
    --title "LEVIR_CD_Experiment_distillation50" \
    --data_name "LEVIR+" \
    --train_dataset_path "../../Dataset/LEVIR-CD+/train/" \
    --train_list_path "../../Dataset/LEVIR-CD+/train.txt" \
    --val_dataset_path "../../Dataset/LEVIR-CD+/val/" \
    --val_list_path "../../Dataset/LEVIR-CD+/val.txt" \
    --test_dataset_path "../../Dataset/LEVIR-CD+/test/" \
    --test_list_path "../../Dataset/LEVIR-CD+/test.txt" \
    --savedir "./result/"\
    --learning_rate 5e-4\
    --epochs 500

    git remote add origin https://github.com/Twenty-Gao/flickcdlight02.git
    git branch -M main
    git push -u origin main

    ssh-keygen -t ed25519 -C "2671628377@qq.com"

    ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIESOGTf2Qg0PFT+BaZekTNde+umY4pJpEv5f9EhM/K6V 2671628377@qq.com

    git remote set-url origin git@github.com:Twenty-Gao/flickcdlight02.git
    git push -u origin main