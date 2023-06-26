python ./ImageAestheticsGANs/acgan_train.py \
  --batch_size 64 \
  --max_epochs 200 \
  --image_size 64 \
  --lrg 0.0002 \
  --lrd 0.0002 \
  --sample_path F:\\Projects\\Disertatie\\RESULTS\\cGAN_64x64

python ./ImageAestheticsGANs/acgan_train.py \
  --batch_size 64  \
  --max_epochs 200 \
  --image_size 128 \
  --lrg 0.0002 \
  --lrd 0.0002 \
  --sample_path F:\\Projects\\Disertatie\\RESULTS\\cGAN_128x128

python ./ImageAestheticsGANs/acgan_train.py \
  --batch_size 32 \
  --max_epochs 100 \
  --image_size 256 \
  --lrg 0.0002 \
  --lrd 0.0002 \
  --sample_path F:\\Projects\\Disertatie\\RESULTS\\cGAN_256x256

python ./ImageAestheticsGANs/acgan_train.py \
  --batch_size 64 \
  --max_epochs 400 \
  --image_size 64 \
  --is_load True \
  --ckpt_path F:\\Projects\\Disertatie\\RESULTS\\cGAN_64x64\\checkpoint_iteration_30999.tar \
  --lrg 0.0002 \
  --lrd 0.0002 \
  --sample_path F:\\Projects\\Disertatie\\RESULTS\\cGAN_64x64

python ./ImageAestheticsGANs/acgan_train.py \
  --batch_size 64  \
  --max_epochs 400 \
  --image_size 128 \
  --is_load True \
  --ckpt_path F:\\Projects\\Disertatie\\RESULTS\\cGAN_128x128\\checkpoint_iteration_30999.tar \
  --lrg 0.0002 \
  --lrd 0.0002 \
  --sample_path F:\\Projects\\Disertatie\\RESULTS\\cGAN_128x128

python ./ImageAestheticsGANs/acgan_train.py \
  --batch_size 32 \
  --max_epochs 200 \
  --image_size 256 \
  --is_load True \
  --ckpt_path F:\\Projects\\Disertatie\\RESULTS\\cGAN_256x256\\checkpoint_iteration_31099.tar \
  --lrg 0.0002 \
  --lrd 0.0002 \
  --sample_path F:\\Projects\\Disertatie\\RESULTS\\cGAN_256x256

python ./ImageAestheticsGANs/acgan_train.py \
  --batch_size 64  \
  --max_epochs 400 \
  --image_size 128 \
  --lrg 0.0001 \
  --lrd 0.0005 \
  --sample_path F:\\Projects\\Disertatie\\RESULTS\\cGAN_128x128_different_lr

python ./ImageAestheticsGANs/acgan_train.py \
  --batch_size 64  \
  --max_epochs 400 \
  --image_size 128 \
  --lrg 0.0002 \
  --lrd 0.0002 \
  --latent_dim 256 \
  --sample_path F:\\Projects\\Disertatie\\RESULTS\\cGAN_128x128

python ./ImageAestheticsGANs/acgan_train.py \
  --batch_size 64  \
  --max_epochs 400 \
  --image_size 128 \
  --n_critic 10 \
  --lrg 0.0002 \
  --lrd 0.0002 \
  --sample_path F:\\Projects\\Disertatie\\RESULTS\\cGAN_128x128_different_n_critic

python ./ImageAestheticsGANs/acgan_train.py \
  --batch_size 128  \
  --max_epochs 800 \
  --image_size 128 \
  --lrg 0.0005 \
  --lrd 0.0001 \
  --sample_path F:\\Projects\\Disertatie\\RESULTS\\Final

python ./ImageAestheticsGANs/acgan_train.py \
  --batch_size 128  \
  --max_epochs 1600 \
  --image_size 128 \
  --is_load True\
  --lrg 0.0005 \
  --lrd 0.0001 \
  --sample_path F:\\Projects\\Disertatie\\RESULTS\\Final \
  --ckpt_path F:\\Projects\\Disertatie\\RESULTS\\Final\\checkpoint_iteration_61599.tar

