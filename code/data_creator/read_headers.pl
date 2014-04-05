#!/usr/bin/perl
my $filename = $ARGV[0];
open(HEADER_FILE, $filename) or die $!;
while(my $line = <HEADER_FILE>) {
	if ($line =~ /^# Reason for admission: (.*)$/) {
		if ($1 == "Myocardial infarction") {print "MI"}
		elsif ($1 == "Healthy control") {print "HEALTHY"} 
		else {print "UNKNOWN"}
	}
}