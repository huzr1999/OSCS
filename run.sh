
for method in OSCS; do
	for score_name in md badacts; do
		for dataset_name in agnews; do
			for poisoner_name in badnets addsent stylebkd synbkd; do
				uv run python main.py \
					--dataset_name "$dataset_name" \
					--poisoner_name "$poisoner_name" \
					--method $method \
					--score_name $score_name \
					--T 20000\
					--num_train_epochs 5 \
					--model_name roberta-base
					# --model_name bert-base-uncased
			done
		done
	done
done