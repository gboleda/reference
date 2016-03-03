#!/usr/bin/perl -w

# assumes output of evaluate_cnn_baseline.pl

$counts_all=0;
$counts_clean=0;
$hits_all=0;
$hits_clean=0;

while (<>) {
    chomp;
    @F=split "[\t ]+",$_;
    $gold = $F[1];
    $model_guess = $F[$#F];

    $counts_all++;
    
    if ($gold > 0) {
	$counts_clean++;
	if ($model_guess eq "right_referent") {
	    $hits_all++;
	    $hits_clean++;
	}
    }
    elsif ($model_guess eq "multiple_referents"
	   ||
	   $model_guess eq "no_referent") {
	$hits_all++;
    }
}

print "total items: ", $counts_all, ", ";
print "total accuracy ",$hits_all/$counts_all,", ";
print "clean items: ", $counts_clean, ", ";
print "clean accuracy: ", $hits_clean/$counts_clean, "\n";
