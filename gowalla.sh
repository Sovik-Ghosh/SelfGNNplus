CUDA_VISIBLE_DEVICES=1 python3.10 main.py --data gowalla --lr 2e-3 --reg 1e-2 --temp 0.1 --ssl_reg 1e-6  --epoch 150  --batch 512 --sslNum 40 --graphNum 3  --gnn_layer 2  --att_layer 1 --test True --testSize 1000 --ssldim 48 --latdim 80 --log_dir gowalla_tcn_80 --model tcn