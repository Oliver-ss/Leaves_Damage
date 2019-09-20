train:
	python3 train.py

table:
	python3 ../../scripts/show_table.py

test:
	python3 get_predictions.py ${MODEL}

PRED = train_log/test/predictions.json
mAP:
	python3 ../../scripts/eval/get_mAP_multiclass.py ${PRED}

THRES = 0.4
show:
	python3 ../../scripts/eval/visualize.py ${THRES} ${PRED}
