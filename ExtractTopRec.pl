#!/usr/bin/perl
#
# extract top 10 recommendations
#

use strict;
use warnings;

# input should be data/mml/test_nr_predictions.csv

my $last_uid = -1;
my @user_rows = ();


while(<>){
	chomp;
	my($uid,$pid,$q) = split /\t/;
	# a new user starts
	if ($uid != $last_uid && @user_rows > 0) {
		my @sorted_rows = sort {$b->[2] <=> $a->[2]} @user_rows;
		# only keep the top 10
		@sorted_rows = @sorted_rows[0..99];
		# write to output
		map {print join("\t",@$_)."\n"} @sorted_rows;
		@user_rows = ([$uid,$pid,$q]);
		$last_uid = $uid;
	} else {
		$last_uid = $uid;
		push @user_rows , [$uid,$pid,$q];
	}
}

my @sorted_rows = sort {$b->[2] <=> $a->[2]} @user_rows;
# only keep the top 10
@sorted_rows = @sorted_rows[0..99];
# write to output
map {print join("\t",@$_)."\n"} @sorted_rows;
