#!/usr/bin/perl -w

open MODELGUESSES,shift;
while (<MODELGUESSES>) {
    chomp;
    @F = split "\t",$_;
    $model_guess_for{$F[0]} = $F[1];
}
close MODELGUESSES;

open GOLDFILE,shift;
while (<GOLDFILE>) {
    chomp;
    @images = split "[\t ]+",$_;
    $re = shift @F;
    $gold_index = shift @F;
    print $re,"\t",$gold_index;
    $gold_referent_count = 0;
    $index_match = 0;
    $i=0;
    while ($i<=$#images) {
	$index = $i+1;
	if (!defined($guess=$model_guess_for{$images[$i]})) {
	    $guess = "NOGUESS";
	}
	if ($guess eq $re) {
	    $gold_referent_count++;
	    if ($index==$gold_index) {
		$index_match = 1;
	    }
	}
	print "\t",$images[$i],"/",$guess;
    }
    $model_output = "wrong_referent";
    if ($gold_referent_count==0) {
	$model_output = "no_referent";
    }
    elsif ($gold_referent_count>1) {
	$model_output = "multiple_referents";
    }
    elsif ($index_match) {
	$model_output = "right_referent";
    }
}

close GOLDFILE;
