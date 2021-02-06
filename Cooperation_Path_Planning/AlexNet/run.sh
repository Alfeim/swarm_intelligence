#shell
python train.py \
	--MODELTYPE NOQUANT \
	--EVALTYPE TEST \
        --EVAL 0 \
        --BATCH 32 \
        --STEP 15000 \
        --IMGPATH ./train/ \
        --VALPATH ./eval/ \
        --TRAINLOGS ./train_logs/ \
        --VALLOGS ./eval_logs/ \
        --WIDTH 10 \
        --HEIGHT 10 \
        --CHANNELS 2 \
        --LRATE 0.001 \
        --RATIO 0.8 \
