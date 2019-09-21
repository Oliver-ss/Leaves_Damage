train:
	python3 train.py

table:
	python3 ../../scripts/show_table.py

test:
	python3 get_predictions.py ${MODEL}

PRED = train_log/test/predictions.json
mAP:
	python3 ../../scripts/eval/get_mAP.py ${PRED}

THRES1 = 0.4
THRES2 = 0.6
THRES3 = 0.5
show:
	python3 ../../scripts/eval/visualize.py ${THRES1} ${THRES2} ${THRES3} ${PRED}
