#!/usr/bin/perl -w

while (<>) {
    chomp;
    ($id,$guesses_string) = split "\t",$_;
    $id =~ s/\.jpg$//;
    $gold = $id;
    $gold =~ s/_.*$//;

    @guesses = split ",",$guesses_string;
    $found_correct_match = 0;
    $picked = "";

    foreach $guess (@guesses) {
	$guess =~ s/^ //;
	$guess = lc($guess);
	if ($guess eq $gold) {
	    $picked = $gold;
	    $found_correct_match = 1;
	}
	elsif ($guess=~/ /) {
	    if ($picked eq "") {
		$picked = $guess;
		$picked =~ s/ /_/g;
	    }
	    @parts = split " ",$guess;
	    foreach $part (@parts) {
		if ($part eq $gold) {
		    $picked = $gold;
		    $found_correct_match = 1;
		    last;
		}
	    }
	}
	elsif ($picked eq "" || $picked =~/_/) {
	    $picked = $guess;
	}
	if ($found_correct_match) {
	    last;
	}
    }
    print $id,"\t",$picked,"\n";
}

