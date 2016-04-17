#!/usr/bin/perl -w

$local_path = shift;

while (<>) {
    chomp;
    @F=split "[ \t]+",$_;
    $id = $F[0];
    $file_path = join "\\ ",@F[1..$#F];
    $file_name = $id . ".jpg";

    `cp $file_path $local_path/$file_name`;
}


