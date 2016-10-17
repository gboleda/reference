#!/usr/bin/perl -w

use Math::Random;

$max_samples = shift;
$original_exposures_count = shift;
$new_exposures_count = $max_samples*$original_exposures_count;

while (<>) {
    chomp;
    @F = split "[\t ]+",$_;
    $index_of_first_exposure=$#F-$original_exposures_count+1;
    @original_exposures = splice @F, $index_of_first_exposure;
    @new_exposures = ();
    foreach $original_exposure (@original_exposures) {
	$sample_count = random_uniform_integer(1,1,3);
	while ($sample_count>0) {
	    push @new_exposures,$original_exposure;
	    $sample_count--;
	}
    }
    while ($#new_exposures<($new_exposures_count-1)) {
	push @new_exposures,"NA:NA";
    }
    print join ("\t",@F),"\t";
    print join "\t",(random_permutation(@new_exposures)),"\n";
}

	
