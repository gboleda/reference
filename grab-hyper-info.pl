#!/usr/bin/perl -w

while (<>) {
    chomp;
    if (/mini_batch_size : ([0-9]+)/) {
	$mini_batch_size = $1;
    }
    elsif (/reference_size : ([0-9]+)/) {
	$reference_size = $1;
    }
    elsif (/momentum : ([0-9\.e\-]+)/) {
	$momentum = $1;
    }
    elsif (/learning_rate : ([0-9\.e\-]+)/) {
	$learning_rate = $1;
    }
    elsif (/done with epoch ([0-9]+) with average training loss ([0-9\.e\-]+)/) {
	$epoch = $1;
	$training_losss = $2
    }
    elsif (/validation loss: ([0-9\.e\-]+)/) {
	$validation_loss = $1;
    }
    elsif (/validation accuracy: ([0-9\.e\-]+)/) {
	$validation_accuracy = $1;
	print join "\t",($mini_batch_size,$reference_size,$momentum,$learning_rate,$epoch,$training_losss,
			 $validation_loss,$validation_accuracy)
    }
}






