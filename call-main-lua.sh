#th -i main.lua --word_lexicon_file word-lexicon-toy.dm --image_dataset_file image-dataset-toy.dm --stimuli_prefix stimuli

# quick test
# th main.lua --toy 1 --t_input_size 100 --training_set_size 500 --reference_size 80 --validation_set_size 100 --mini_batch_size 2 --normalize_embeddings 1 --image_set_size 5 --min_filled_image_set_size 2 --max_epochs 10

#th main.lua --model ff_ref --training_set_size 10000 --validation_set_size 3000 --test_set_size 0 --image_set_size 10 --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --protocol_prefix $datadir/stimuli --max_epochs 17 --min_epochs 17 --normalize_embeddings 1 --reference_size 300 --learning_rate 0.09 --momentum 0.09 --mini_batch_size 1 --save_model_to_file chosen_file.bin

# datadir='/Users/gboleda/Desktop/data/reference/exp5-upto5-tiny'
# th main.lua --model ff_ref_deviance --training_set_size 50 --validation_set_size 50 --test_set_size 50 --image_set_size 125 --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --protocol_prefix $datadir/stimuli.deviant --max_epochs 17 --min_epochs 17 --normalize_embeddings 1 --reference_size 300 --learning_rate 0.09 --momentum 0.09 --mini_batch_size 1

# datadir='/Users/gboleda/Desktop/data/reference/exp5-upto5-tiny'
# th main.lua --model ff_ref_deviance --training_set_size 50 --validation_set_size 50 --test_set_size 50 --image_set_size 125 --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --protocol_prefix $datadir/stimuli.deviant --max_epochs 1 --min_epochs 1 --normalize_embeddings 1 --reference_size 300 --learning_rate 0.09 --momentum 0.09 --mini_batch_size 1

# th max-margin-test.lua

datadir='/Users/gboleda/Desktop/data/reference/exp5-upto5-tiny-nodeviants'
# th /Users/gboleda/Desktop/tmp/reference/main.lua
# th main.lua --model ff_ref --training_set_size 40 --validation_set_size 40 --test_set_size 30 --image_set_size 125 --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --protocol_prefix $datadir/stimuli --max_epochs 1 --min_epochs 1 --normalize_embeddings 1 --reference_size 300 --learning_rate 0.09 --momentum 0.09 --mini_batch_size 1

th main.lua --model ff_ref --training_set_size 5 --validation_set_size 5 --test_set_size 5 --image_set_size 125 --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --protocol_prefix $datadir/stimuli-tiny --max_epochs 1 --min_epochs 1 --normalize_embeddings 1 --reference_size 300 --learning_rate 0.09 --momentum 0.09 --mini_batch_size 1

# datadir='/Users/gboleda/Desktop/data/reference/exp5-upto5-tiny-nodeviants'
th main.lua --model max_margin_bl --training_set_size 40 --validation_set_size 40 --test_set_size 30 --image_set_size 125 --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --protocol_prefix $datadir/stimuli --max_epochs 1 --min_epochs 1 --normalize_embeddings 1 --reference_size 300 --learning_rate 0.09 --momentum 0.09 --mini_batch_size 1
