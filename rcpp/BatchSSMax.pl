#/usr/bin/perl
# run SSMax with varying regularization values

use strict;
use warnings;

my @gamma_vals = (0.1,1,2,5,10,20);

my $maxEpochs = 100;
my $costPercentage = 0.5;

foreach my $gamma (@gamma_vals) { 
	my $cmd = "./SSMax ../data/1015/train.csv ../data/1015/test.csv ../data/1016 $gamma $maxEpochs $costPercentage";
	print $cmd . "\n";
	`$cmd`; 
}