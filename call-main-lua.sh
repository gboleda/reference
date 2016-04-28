#th -i main.lua --word_lexicon_file word-lexicon-toy.dm --image_dataset_file image-dataset-toy.dm --stimuli_prefix stimuli

# quick test
# th main.lua --toy 1 --t_input_size 100 --training_set_size 500 --reference_size 80 --validation_set_size 100 --mini_batch_size 2 --normalize_embeddings 1 --image_set_size 5 --min_filled_image_set_size 2 --max_epochs 10

#th main.lua --model ff_ref --training_set_size 10000 --validation_set_size 3000 --test_set_size 0 --image_set_size 10 --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --protocol_prefix $datadir/stimuli --max_epochs 17 --min_epochs 17 --normalize_embeddings 1 --reference_size 300 --learning_rate 0.09 --momentum 0.09 --mini_batch_size 1 --save_model_to_file chosen_file.bin

# datadir='/Users/gboleda/Desktop/data/reference/exp5-upto5-tiny'
# th main.lua --model ff_ref_deviance --training_set_size 50 --validation_set_size 50 --test_set_size 50 --image_set_size 125 --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --protocol_prefix $datadir/stimuli.deviant --max_epochs 17 --min_epochs 17 --normalize_embeddings 1 --reference_size 300 --learning_rate 0.09 --momentum 0.09 --mini_batch_size 1

# datadir='/Users/gboleda/Desktop/data/reference/exp5-upto5-tiny'
# th main.lua --model ff_ref_deviance --training_set_size 50 --validation_set_size 50 --test_set_size 50 --image_set_size 125 --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --protocol_prefix $datadir/stimuli.deviant --max_epochs 1 --min_epochs 1 --normalize_embeddings 1 --reference_size 300 --learning_rate 0.09 --momentum 0.09 --mini_batch_size 1

# th max-margin-test.lua

# -----

date

margin=0.1
max_epochs=1
min_epochs=1
mini_batch_size=1

# -- tiny dataset

datadir='/Users/gboleda/Desktop/love-project/data/exp5-upto5-tiny-nodeviants'

# th main.lua --model ff_ref --training_set_size 40 --validation_set_size 40 --test_set_size 30 --image_set_size 125 --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --protocol_prefix $datadir/stimuli --max_epochs 1 --min_epochs 1 --normalize_embeddings 1 --reference_size 300 --learning_rate 0.09 --momentum 0.09 --mini_batch_size $mini_batch_size

# th main.lua --model ff_ref --training_set_size 5 --validation_set_size 5 --test_set_size 5 --image_set_size 125 --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --protocol_prefix $datadir/stimuli-tiny --max_epochs 1 --min_epochs 1 --normalize_embeddings 1 --reference_size 300 --learning_rate 0.09 --momentum 0.09 --mini_batch_size $mini_batch_size

th main.lua --model max_margin_bl --margin $margin --training_set_size 40 --validation_set_size 40 --test_set_size 30 --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --protocol_prefix $datadir/stimuli --max_epochs $max_epochs --min_epochs $min_epochs --normalize_embeddings 1 --reference_size 300 --learning_rate 0.09 --momentum 0.09 --mini_batch_size $mini_batch_size

#th main.lua --model max_margin_bl --margin $margin --training_set_size 5 --validation_set_size 5 --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --protocol_prefix $datadir/stimuli-tiny --max_epochs $max_epochs --min_epochs $min_epochs --normalize_embeddings 1 --reference_size 300 --learning_rate 0.09 --momentum 0.09 --mini_batch_size $mini_batch_size

# # -- 10K-3K-5K, no deviants
# datadir='/Users/gboleda/Desktop/data/reference/new-exp3a'
# th main.lua --model max_margin_bl --margin $margin --training_set_size 10000 --validation_set_size 3000 --test_set_size 5000 --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --protocol_prefix $datadir/stimuli --max_epochs $max_epochs --min_epochs $min_epochs --normalize_embeddings 1 --reference_size 300 --learning_rate 0.09 --momentum 0.09 --mini_batch_size $mini_batch_size --save_model_to_file new-exp3a-max-margin-bl-margin01.model

# -- new-exp3, training on no deviant portion
# datadir='/Users/gboleda/Desktop/data/reference/new-exp3'
# th main.lua --model max_margin_bl --margin $margin --training_set_size 27981 --validation_set_size 3503 --test_set_size 6978 --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --protocol_prefix $datadir/stimuli-nodevs --max_epochs $max_epochs --min_epochs $min_epochs --normalize_embeddings 1 --reference_size 300 --learning_rate 0.09 --momentum 0.09 --mini_batch_size $mini_batch_size --save_model_to_file new-exp3-max-margin-bl.model

date
