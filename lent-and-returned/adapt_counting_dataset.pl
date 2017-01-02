#!/usr/bin/perl -w

#ex:
# cephalopod	3	NA_NA	grocer_103933	grocer_103201	NA_NA	NA_NA	NA_NA	cephalopod_172475	grocer_103933	NA_NA	grocer_103933	troopship_465411	cephalopod_172475	cephalopod_172475	cephalopod_172898	NA_NA	troopship_465308	NA_NA	cephalopod_172898	cephalopod_172475	cephalopod_172898	grocer_103201	cephalopod_172216	NA_NA	grocer_103933	manatee_272876	troopship_465411	grocer_102941	cephalopod_172898	NA_NA	NA_NA
# to
# dryer:put:recommend	6	||	material_294545	dryer_347467	material_294276	material_294076	dryer_348324	dryer_348003	||	keep:material_294076	keep:material_294276	put:dryer_348003	recommend:dryer_348324	recommend:dryer_348003	keep:dryer_348324	put:dryer_347467	put:material_294276	keep:dryer_347467	recommend:material_294545	put:material_294545	recommend:material_294076
#

while (<>) {
    chomp;
    @F = split "[\t ]+",$_;
    print $F[0], ":NA:NA\t";
    print "||\t||";
    foreach my $i (1..$#F){
	print "\tNA:$F[$i]";
    }
    print "\n";
}
