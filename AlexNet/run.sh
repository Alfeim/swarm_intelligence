#shell
python train.py \
	--MODELTYPE NOQUANT \
	--EVALTYPE TEST \
        --EVAL 1 \
        --BATCH 10 \
        --STEP 20000 \
        --IMGPATH ./train/ \
        --VALPATH ./eval/ \
        --TRAINLOGS ./train_logs/ \
        --VALLOGS ./eval_logs/ \
        --WIDTH 10 \
        --HEIGHT 10 \
        --CHANNELS 2 \
        --LRATE 0.005 \
        --RATIO 0.8 \
