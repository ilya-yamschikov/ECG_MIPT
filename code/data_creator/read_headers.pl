#!/usr/bin/perl

my %REASONS = (
	"Myocardial infarction" => {
		class => "MI",
		include => 1
	},
	"Cardiomyopathy" => {
		class => "CARDIOMYOPATHY",
		include => 0
	},
	"Healthy control" => {
		class => "HEALTHY",
		include => 1
	}
);

my $filename = $ARGV[0];
if (defined $ARGV[1]) {
	if ($ARGV[1] eq "-c") {
		$mode = "class"
	} elsif ($ARGV[1] eq "-i") {
		$mode = "include"
	} else {
		die "unknown argument";
	}
} else {
	$mode = "class"
}

open(HEADER_FILE, $filename) or die $!;
while(my $line = <HEADER_FILE>) {
	if ($line =~ /^# Reason for admission: (.*)$/) {
		my $reason = $1;
		$reason =~ s/\R//g;
		if (exists $REASONS{$reason}) {
			print $REASONS{$reason}{$mode}
		}
		else {
			if ($mode eq "include") {print 0}
			else {print "UNKNOWN"}
		}
	}
}