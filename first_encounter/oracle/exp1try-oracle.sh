# DATA
datadir='/home/nghia/marco/reference/association/exp-try1'
trsetsize=40000
vsetsize=5000
tesetsize=10000
prefixfortraining=stimuli
test_file=stimuli.test

# MODEL PARAMETERS
model=entity_prediction_image_oracle
mini_batch_size=10 # consider 1, 10, 20, 100 future
lr_decay=0.0001 # values for future: 0.001 0.0001
momentum=0.09 # values for future: 0.0, 0.09*, 0.3, 0.6, 0.9
learning_rate=0.09
input_sequence_cardinality=12
new_mass_aggregation_method=max
new_cell_nonlinearity=sigmoid
multimodal_size=600
summary_size=300
hidden_size=300
hidden_count=2 # number of hidden layers [1]
ff_nonlinearity=sigmoid # nonlinear transformation of hidden layers (options: none (default), sigmoid, relu, tanh) [none]
dropout_p=0.5
cuda=1
min_epochs=5
max_epochs=10
max_validation_lull=5

# OUTPUT PARAMETERS
prefixoutput=$prefixfortraining-longepoch-exp-bal12-$model-mm$multimodal_size-lr$learning_rate-batch$mini_batch_size
modelfile=$prefixoutput/$prefixoutput.bin
guessesfile=$prefixoutput/$prefixoutput.guesses
test_debugprefix=$prefixoutput/$prefixoutput.test.debug
train_debugprefix=$prefixoutput/$prefixoutput.train.debug

# choose GPU
export CUDA_VISIBLE_DEVICES=0

date
mkdir $prefixoutput

th train.lua --model $model --training_set_size $trsetsize --validation_set_size $vsetsize --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --protocol_prefix $datadir/$prefixfortraining --normalize_embeddings 1 --learning_rate $learning_rate --momentum $momentum --mini_batch_size $mini_batch_size --input_sequence_cardinality $input_sequence_cardinality --dropout_prob $dropout_p --multimodal_size $multimodal_size --summary_size $summary_size --hidden_size $hidden_size --use_cuda $cuda --new_mass_aggregation_method $new_mass_aggregation_method --min_epochs $min_epochs --max_epochs $max_epochs --max_validation_lull $max_validation_lull --output_debug_prefix $train_debugprefix --output_guesses_file $guessesfile --save_model_to_file $modelfile --new_cell_nonlinearity $new_cell_nonlinearity


date

#echo "test.lua --test_set_size $tesetsize --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --test_file $datadir/$test_file --output_debug_prefix $debugprefix --normalize_embeddings 1 --input_sequence_cardinality $input_sequence_cardinality --model_file $modelfile --output_guesses_file $guessesfile --use_cuda $cuda"
th test.lua --test_set_size $tesetsize --test_file $datadir/$test_file --output_debug_prefix $test_debugprefix --normalize_embeddings 1 --input_sequence_cardinality $input_sequence_cardinality --model_file $modelfile --output_guesses_file $guessesfile --use_cuda $cuda 
