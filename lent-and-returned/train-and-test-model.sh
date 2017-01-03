date

# DATA
datadir='exp-bal12'
trsetsize=40000
vsetsize=5000
tesetsize=10000
prefixfortraining=stimuli
test_file=stimuli.test

# MODEL PARAMETERS
model=entity_prediction_image_att_shared
mini_batch_size=10 # consider 1, 10, 20, 100 future
lr_decay=0.0001 # values for future: 0.001 0.0001
momentum=0.09 # values for future: 0.0, 0.09*, 0.3, 0.6, 0.9
learning_rate=0.09
input_sequence_cardinality=36
candidate_cardinality=6
new_mass_aggregation_method=max
multimodal_size=1000
summary_size=300
hidden_size=300
hidden_count=2 # number of hidden layers [1]
ff_nonlinearity=sigmoid # nonlinear transformation of hidden layers (options: none (default), sigmoid, relu, tanh) [none]
dropout_p=0.5
cuda=1
min_epochs=5
max_epochs=500
max_validation_lull=5

# OUTPUT PARAMETERS
prefixoutput=multiple_exp12_mm_1000_lr_0.09_batch_10
modelfile=$prefixoutput.bin
guessesfile=$prefixoutput.guesses
test_debugprefix=$prefixoutput.test.debug
#train_debugprefix=$prefixoutput.train.debug

# choose GPU
export CUDA_VISIBLE_DEVICES=0

date

echo datadir: $datadir

th lent-and-returned-train.lua --model $model --training_set_size $trsetsize --validation_set_size $vsetsize --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --protocol_prefix $datadir/$prefixfortraining --normalize_embeddings 1 --learning_rate $learning_rate --momentum $momentum --mini_batch_size $mini_batch_size --input_sequence_cardinality $input_sequence_cardinality --candidate_cardinality $candidate_cardinality --dropout_prob $dropout_p --multimodal_size $multimodal_size --summary_size $summary_size --hidden_size $hidden_size --use_cuda $cuda --new_mass_aggregation_method $new_mass_aggregation_method --min_epochs $min_epochs --max_epochs $max_epochs --max_validation_lull $max_validation_lull --output_debug_prefix $train_debugprefix --save_model_to_file $modelfile

# th lent-and-returned-test.lua --test_set_size $tesetsize --test_file $datadir/$test_file --input_sequence_cardinality $input_sequence_cardinality --model_file $modelfile --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --normalize_embeddings 1 --output_guesses_file $output_guesses_file

date

# best parameters from marco
# th lent-and-returned-train.lua --protocol_prefix  /home/sebastian.pado/binding/exp1-large/stimuli --word_embedding_file 
# /home/sebastian.pado/binding/exp1-large/word.dm  --image_embedding_file /home/sebastian.pado/binding/exp1-large/image.dm
#  --normalize_embeddings 1 --input_sequence_cardinality 6 --training_set_size 40000 --validation_set_size 5000 --multimod
# al_size 300 --mini_batch_size 1 --learning_rate 0.09 --momentum 0.09 --learning_rate_decay 0.0001 --save_model_to_file t
# olerant-exp1-large.bin --new_mass_aggregation_method sum --max_validation_lull 5
