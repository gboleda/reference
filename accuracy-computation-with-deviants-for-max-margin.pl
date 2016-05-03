#!/usr/bin/perl -w

use List::Util max;

$zerothreshold = shift;
$multithreshold = shift;

# Example:
# 0 5 0.027808980226855 -0.026260225584297 -0.027621299657854 -0.060993691776221 0.013054225188681 0.027808980226855
# 2 2 0.044788000246237 -0.07754712280129 0.044788000246237
# 1 2 0.076732558704314 0.063619253827331 0.076732558704314

$thitcount=0; $phitcount=0; $mirhitcount=0; $murhitcount=0;
$tcount=0; $pcount=0; $mircount=0; $murcount=0; $pdevcount=0;
while(<>){
    print $_;
    # chomp;
    # print "$_: ";
    # ($gold, $pred, $max, @PREDS) = split(" ", $_);
    # $tcount++;
    # # by default, gold and pred are as they are; changed if they are deviant
    # $ngold=$gold;
    # $npred=$pred;
    # $toprintpred=$pred; # this is for printing purposes (different than pred for accuracy computation)
    # print "$gold $pred";
    # print "\t---\t"; 
    # # print join(" ",@PREDS);
    # if($gold==0 or $gold==-1){$ngold=6};
    # if ($max < $zerothreshold){$npred=6;$toprintpred=0};
    # @SORTED=sort(@PREDS);
    # $max2=$SORTED[-2];
    # $maxdiff=$max-$max2;
    # print "$max $max2";
    # print "\t---\t"; 
    # print join(" ",@SORTED);
    # if ($maxdiff < $multithreshold){$npred=6;$toprintpred=-1};
    # if($ngold==$npred){$thitcount++;}
    # if($gold==0){$mircount++; if($ngold==$npred){$mirhitcount++;}}
    # elsif($gold==-1){$murcount++; if($ngold==$npred){$murhitcount++;}}
    # else{
    # 	$pcount++; 
    # 	if($ngold==$npred){$phitcount++;}
    # 	elsif($npred==6){$pdevcount++}
    # }
    # #print join(" ",($gold, $pred, $toprintpred, @PREDS));
    # print "\n";
}
$taccuracy=$thitcount/$tcount;
$paccuracy=$phitcount/$pcount;
$miraccuracy=$mirhitcount/$mircount;
$muraccuracy=$murhitcount/$murcount;
$wrongdeviants=$pdevcount/$pcount;
# print STDERR "THRESHOLDS: MISSING REF = $zerothreshold MULTI REF = $multithreshold\n";
# print STDERR "TOTAL ACCURACY: $taccuracy\n";
# print STDERR "POINTING ACCURACY: $paccuracy; PREDICTED TO BE DEVIANT: $wrongdeviants\n";
# print STDERR "MISSREF ACCURACY: $miraccuracy\n";
# print STDERR "MULTREF ACCURACY: $muraccuracy\n";
# print STDERR "$zerothreshold $multithreshold $taccuracy\n";
print STDERR "$zerothreshold $multithreshold $taccuracy\n";
