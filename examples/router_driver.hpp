#ifndef MITM_ROUTER_DRIVER
#define MITM_ROUTER_DRIVER

#include <getopt.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <err.h>

#include "router/router.hpp"

/*
 * The command line shared by router_test and router_bench: the team, the traffic, every Router_Opts
 * field.  Part of the examples, not of the library.
 */
namespace mitm {

struct RouterArgs {
	int producers = 4;                 /* per node */
	int consumers = 2;               /* per node */
	u64 points = 100000;             /* per producer and per round (test) */
	int rounds = 3;
	Router_Opts opts;
	int skew = 0;                    /* hot destinations; 0 == uniform */
	bool stagger = false;            /* producer s pushes (s+1)/S of the points */
	bool partial = false;            /* 1..7 points per destination, round robin */
	bool slow_recv = false;          /* a consumer dawdles after each pop */
	bool check_limit = false;        /* test: the consumers are the limit -- the producers wait, the verdict says so */
	u64 seed = 1;                    /* bench: the producers' PRNG streams */
	double seconds = 3;              /* bench: duration of a round */
	bool local_only = false;         /* bench: every destination on the producer's node */
	std::string test = "all";
};

static void router_usage()
{
	printf("Router drivers.  Options:\n");
	printf("  --producers N --consumers N        threads per node (default 4 and 2)\n");
	printf("  --points N --rounds N            per producer per round; rounds\n");
	printf("  --block N --swc N --n-recv N --inbox BLOCKS --credit N\n");
	printf("  --skew K --stagger --partial --slow-recv --seed S\n");
	printf("  --no-bind                        do not pin the threads (on by default)\n");
	printf("  --seconds X --local-only               (bench)\n");
	printf("  --test NAME                      (test) placement connect_only basic tiny rounds zero skew\n");
	printf("                                   partial_lines swc_is_block staggered_close slow_recv\n");
	printf("                                   asymmetric limits all\n");
	printf("  --quiet\n");
	exit(EXIT_SUCCESS);
}

static void router_parse(int argc, char **argv, RouterArgs &a)
{
	enum {
		O_PRODUCERS = 1000, O_CONSUMERS, O_POINTS, O_ROUNDS, O_BLOCK, O_SWC, O_NRECV, O_INBOX,
		O_CREDIT, O_SKEW, O_STAGGER, O_PARTIAL, O_SLOW,
		O_SEED, O_SECONDS, O_NOBIND, O_LOCAL, O_TEST, O_QUIET, O_HELP
	};
	struct option longopts[] = {
		{"producers", required_argument, NULL, O_PRODUCERS}, {"consumers", required_argument, NULL, O_CONSUMERS},
		{"points", required_argument, NULL, O_POINTS}, {"rounds", required_argument, NULL, O_ROUNDS},
		{"block", required_argument, NULL, O_BLOCK}, {"swc", required_argument, NULL, O_SWC},
		{"n-recv", required_argument, NULL, O_NRECV}, {"inbox", required_argument, NULL, O_INBOX},
		{"credit", required_argument, NULL, O_CREDIT},
		{"skew", required_argument, NULL, O_SKEW},
		{"stagger", no_argument, NULL, O_STAGGER}, {"partial", no_argument, NULL, O_PARTIAL},
		{"slow-recv", no_argument, NULL, O_SLOW}, {"seed", required_argument, NULL, O_SEED},
		{"seconds", required_argument, NULL, O_SECONDS},
		{"no-bind", no_argument, NULL, O_NOBIND},
		{"local-only", no_argument, NULL, O_LOCAL}, {"test", required_argument, NULL, O_TEST},
		{"quiet", no_argument, NULL, O_QUIET}, {"help", no_argument, NULL, O_HELP},
		{NULL, 0, NULL, 0}
	};
	for (;;) {
		int ch = getopt_long(argc, argv, "", longopts, NULL);
		switch (ch) {
		case -1: return;
		case O_PRODUCERS: a.producers = atoi(optarg); break;
		case O_CONSUMERS: a.consumers = atoi(optarg); break;
		case O_POINTS: a.points = human_parse(optarg); break;
		case O_ROUNDS: a.rounds = atoi(optarg); break;
		case O_BLOCK: a.opts.block_points = human_parse(optarg); break;
		case O_SWC: a.opts.swc_linesize = human_parse(optarg); break;
		case O_NRECV: a.opts.n_recv = atoi(optarg); break;
		case O_INBOX: a.opts.inbox_blocks = atoi(optarg); break;
		case O_CREDIT: a.opts.credit = atoi(optarg); break;
		case O_SKEW: a.skew = atoi(optarg); break;
		case O_STAGGER: a.stagger = true; break;
		case O_PARTIAL: a.partial = true; break;
		case O_SLOW: a.slow_recv = true; break;
		case O_SEED: a.seed = strtoull(optarg, NULL, 0); break;
		case O_SECONDS: a.seconds = atof(optarg); break;
		case O_NOBIND: a.opts.pin = false; break;
		case O_LOCAL: a.local_only = true; break;
		case O_TEST: a.test = optarg; break;
		case O_QUIET: a.opts.verbose = false; break;
		case O_HELP: router_usage(); break;
		default: errx(1, "Unknown option (try --help)");
		}
	}
}

}
#endif
