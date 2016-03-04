#!/usr/bin/perl -w

open ILSVRC,shift;
while (<ILSVRC>) {
    chomp;
    s/,//g;
    @words = split "[\t ]+",lc($_);
    foreach $word (@words) {
	$in_ilsvrc{$word}=1;
    }
}
close ILSVRC;

open VOCABULARY,shift;
while (<VOCABULARY>) {
    chomp;
    $word = $_;
    $coverage = 0;
    if ($in_ilsvrc{$word}) {
	$coverage = 1;
    }
    print $word,"\t",$coverage,"\n";
}
close VOCABULARY;
