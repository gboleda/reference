date
#datadir='/Users/gboleda/Desktop/data/reference/new-exp3a'
expdir='/home/boledatorrent/reference'
datadir=$expdir'/data/new-exp3a'
resultsdir=$expdir'/results'
echo datadir: $datadir

max_epochs=20 # Paper: 1 ... 14* ... 50 (for mini-batch sizes >1, epochs up to mini-batch size x 15)
min_epochs=8
mini_batch_size=1 # consider 10, 20, 100 future
lr_decay=0.0001 # values for future: 0.001 0.0001
momentum=0.09 # values for future: 0.0, 0.09*, 0.3, 0.6, 0.9
trainsize=10000
validsize=3000
testsize=5000

for margin in 0.5 # 0.1 0.5 1
do
    for ref_size in 300 # 200
    do
	for learning_rate in 0.09 # 0.03 0.009
	do
	date
	echo CURRENT PARAMETERS: $margin $ref_size $learning_rate 
	th main.lua --model max_margin_bl --margin $margin --training_set_size $trainsize --validation_set_size $validsize --test_set_size $testsize --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --protocol_prefix $datadir/stimuli --max_epochs $max_epochs --min_epochs $min_epochs --normalize_embeddings 1 --reference_size $ref_size --learning_rate $learning_rate --momentum $momentum --learning_rate_decay $lr_decay --mini_batch_size $mini_batch_size --save_model_to_file new-exp3a-max-margin-$margin-$ref_size-$learning_rate.model
	done
    done
done
date
