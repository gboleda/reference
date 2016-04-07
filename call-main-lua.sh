#th -i main.lua --word_lexicon_file word-lexicon-toy.dm --image_dataset_file image-dataset-toy.dm --stimuli_prefix stimuli


# quick test
# th main.lua --toy 1 --t_input_size 100 --training_set_size 500 --reference_size 80 --validation_set_size 100 --mini_batch_size 2 --normalize_embeddings 1 --image_set_size 5 --min_filled_image_set_size 2 --max_epochs 10

datadir='/Users/gboleda/Desktop/data/reference/exp3-upto5'

#th main.lua --model ff_ref --training_set_size 10000 --validation_set_size 3000 --test_set_size 0 --image_set_size 10 --word_embedding_file $datadir/word.dm --image_embedding_file $datadir/image.dm --protocol_prefix $datadir/stimuli --max_epochs 17 --min_epochs 17 --normalize_embeddings 1 --reference_size 300 --learning_rate 0.09 --momentum 0.09 --mini_batch_size 1 --save_model_to_file chosen_file.bin

th main.lua --model ff_ref --training_set_size 10 --validation_set_size 10 --test_set_size 10 --image_set_size 10 --word_embedding_file $datadir/toy-word.dm --image_embedding_file $datadir/toy-image.dm --protocol_prefix $datadir/toy-stimuli --max_epochs 17 --min_epochs 17 --normalize_embeddings 1 --reference_size 300 --learning_rate 0.09 --momentum 0.09 --mini_batch_size 1 --save_model_to_file chosen_file.bin


