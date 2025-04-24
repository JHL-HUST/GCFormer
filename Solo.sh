


acm

python train_test.py --beta=0.1 --attention_dropout=0.1 --alpha=0.7 --dataset=acm --dropout=0.5 \
       --hidden=64 --n_heads=1 --n_layers=1 --peak_lr=0.001 --pp_k=2 --sample_num_n=3 \
       --sample_num_p=7 --temp=0.5 --weight_decay=1e-05 --batch_size 5000


photo

python train_test.py --beta=0.1 --attention_dropout=0.1 --alpha=0.3 --dataset=photo --dropout=0.5 \
                     --hidden=128 --n_heads=1 --n_layers=1 --peak_lr=0.0005 --pp_k=2 --sample_num_n=5 --sample_num_p=3 \
                     --temp=5 --weight_decay=1e-05 --device 2 --batch_size 5000



computer

python train_test.py --beta=0.1 --attention_dropout=0.1 --alpha=0.3 --dataset=computer --dropout=0.1 \
                     --hidden=128 --n_heads=1 --n_layers=1 --peak_lr=0.0005 --pp_k=2 --sample_num_n=5 --sample_num_p=3 \
                     --temp=0.5 --weight_decay=1e-05 --device 0  --batch_size 5000



BlogCatalog

python train_test.py --beta=0.1 --attention_dropout=0.5 --alpha=0.5 --dataset=BlogCatalog --dropout=0.5 \
                     --hidden=128 --n_heads=1 --n_layers=1 --peak_lr=0.001 --pp_k=2 --sample_num_n=7 --sample_num_p=5 \
                     --temp=10 --weight_decay=1e-05 --device 2 





flickr

python train_test.py --beta=0.1 --attention_dropout=0.5 --alpha=0.5 --dataset=flickr --dropout=0.5 \
                     --hidden=128 --n_heads=1 --n_layers=1 --peak_lr=0.0005 --pp_k=2 --sample_num_n=5 --sample_num_p=3 \
                     --temp=5 --weight_decay=1e-05 --device 3 



uai

python train_test.py --beta=0.9 --attention_dropout=0.3 --alpha=0.7 --dataset=uai --dropout=0.5 \
                     --hidden=512 --n_heads=1 --n_layers=1 --peak_lr=0.001 --pp_k=2 --sample_num_n=7 --sample_num_p=5 \
                     --temp=2 --weight_decay=1e-05 --device 2 




corafull

python train_test.py --beta=0.01 --attention_dropout=0.5 --alpha=0.25 --dataset=corafull --dropout=0.3 \
                     --hidden=512 --n_heads=1 --n_layers=1 --peak_lr=0.0005 --pp_k=3 --sample_num_n=7 --sample_num_p=7 \
                     --temp=3 --weight_decay=1e-05 --device 2 


romanempire

python train_test.py --beta=0.01 --attention_dropout=0.5 --alpha=0.4 --dataset=romanempire --dropout=0.5 \
                     --hidden=128 --n_heads=1 --n_layers=1 --peak_lr=0.001 --pp_k=1 --sample_num_n=7 --sample_num_p=7 \
                     --temp=3 --weight_decay=1e-05 --device 0 




python=3.8
cuda=10.2
torch=1.11
numpy=1.22.4
torch-cluster=1.6.0
torch-geometric=2.3.1                  
torch-scatter=2.0.9                 
torch-sparse=0.6.13 
torch-geometric=2.3.1