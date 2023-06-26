python ./ImageAestheticsGANs/resnet18_train.py \
  --batch_size 64 \
  --image_size 256 \
  --results F:\\Projects\\Disertatie\\RESULTS\\resnet18_classification\\standard \
  --lr 0.0002 \
  --optim sgd \
  --criterion bcelogits

python ./ImageAestheticsGANs/resnet18_train.py \
  --batch_size 128 \
  --image_size 256 \
  --results F:\\Projects\\Disertatie\\RESULTS\\resnet18_classification\\batch_size_up \
  --lr 0.0002 \
  --optim sgd \
  --criterion bcelogits

python ./ImageAestheticsGANs/resnet18_train.py \
  --batch_size 64 \
  --image_size 256 \
  --results F:\\Projects\\Disertatie\\RESULTS\\resnet18_classification\\lr_up \
  --lr 0.002 \
  --optim sgd \
  --criterion bcelogits

python ./ImageAestheticsGANs/resnet18_train.py \
  --batch_size 64 \
  --image_size 256 \
  --results F:\\Projects\\Disertatie\\RESULTS\\resnet18_classification\\lr_down \
  --lr 0.00002 \
  --optim sgd \
  --criterion bcelogits

python ./ImageAestheticsGANs/resnet18_train.py \
  --batch_size 64 \
  --image_size 256 \
  --results F:\\Projects\\Disertatie\\RESULTS\\resnet18_classification\\adam \
  --lr 0.0002 \
  --optim adam \
  --criterion bcelogits

python ./ImageAestheticsGANs/resnet18_train.py \
  --batch_size 64 \
  --image_size 256 \
  --results F:\\Projects\\Disertatie\\RESULTS\\resnet18_classification\\cross \
  --lr 0.0002 \
  --optim sgd \
  --criterion cross

python ./ImageAestheticsGANs/resnet18_train.py \
  --batch_size 64 \
  --image_size 256 \
  --results F:\\Projects\\Disertatie\\RESULTS\\resnet18_classification\\focal \
  --lr 0.0002 \
  --optim sgd \
  --criterion focal

# Final
python ./ImageAestheticsGANs/resnet18_train.py \
  --batch_size 128 \
  --image_size 128 \
  --results F:\\Projects\\Disertatie\\RESULTS\\Final_resnet \
  --lr 0.0002 \
  --optim sgd \
  --criterion bcelogits
