#include <assert.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

#include "iimc.h"

struct token_buffer {
	int *buf;
	int max_seq_len;
	int buffer_count;
	int eot_pos;
	int last_pos;
	float oversize_r;
};

static struct token_buffer *token_buffer_new(int max_seq_len,
		float oversize_r)
{
	assert(max_seq_len > 1);
	assert(!(oversize_r < 1.0f));
	assert(!(oversize_r > 3.0f));

	struct token_buffer *b = malloc(sizeof(struct token_buffer));
	if (b == NULL)
		return NULL;

	b->max_seq_len = max_seq_len;
	b->oversize_r = oversize_r;

	b->buffer_count = b->max_seq_len * b->oversize_r + 1;
	b->eot_pos = 0;
	b->last_pos = 0;

	b->buf = malloc(b->buffer_count * sizeof(int));
	if (b->buf == NULL) {
		free(b);
		return NULL;
	}

	b->buf[b->eot_pos] = GPT2_EOT;
	return b;
}

static int token_buffer_free(struct token_buffer *p)
{
	if (p == NULL)
		return IIMC_ENULL_POINTER_FREE;
	if (p->buf != NULL)
		free(p->buf);
	memset(p, 0, sizeof(struct token_buffer));
	free(p);
	return IIMC_ENONE;
}

static int *token_buffer_step(struct token_buffer *b, int *indx)
{
	b->last_pos++;

	if (b->last_pos >= b->buffer_count) {
		memmove(b->buf, &b->buf[b->eot_pos + 1],
				(b->max_seq_len - 1) * sizeof(int));
		b->eot_pos = 0;
		b->last_pos = b->max_seq_len - 1;
	}

	*indx = b->last_pos;
	if (b->last_pos - b->eot_pos >= b->max_seq_len) {
		b->eot_pos = b->last_pos - b->max_seq_len + 1;
		*indx = b->max_seq_len - 1;
	}

	b->buf[b->eot_pos] = GPT2_EOT;
	return &b->buf[b->eot_pos];
}

static void token_buffer_update(struct token_buffer *b, int value)
{
	b->buf[b->last_pos] = value;
}

struct iimc_cfg {
	const char *mf;
	int num_token;
	unsigned long long rng_state;
	char *prompt;
	float oversize_r;
	int seq_len;
};

static void iimc_cfg_default(struct iimc_cfg *p)
{
	p->mf = "gpt2_124M.bin";
	p->num_token = -1;
	p->rng_state = 1337;
	p->prompt = NULL;
	p->oversize_r = 2.0f;
	p->seq_len = -1;
}

static void print_help()
{
	 printf("Usage: iimc [OPTION]... \n"
		"Run inference for GPT2 model to standard output.\n\n"
		"  -h\t\tdisplay this help and exit\n"
		"  -l\t\tlimit the maximum sequence length\n"
		"    \t\tThe limit must be less than the model maximum sequence length.\n"
		"  -m\t\tset model file path\n"
		"  -n\t\tgenerate up to n tokens\n"
		"    \t\tThe number of generated tokens can be larger than the"
		" model maximum\n\t\tsequence length. In that case, the first"
		" tokens are omitted to add\n    \t\tnew tokens at the end.\n"
		"  -r\t\tset buffer oversize ratio\n"
		"    \t\tExtend the token buffer between 1.0 and 3.0 times"
		" the maximum model\n  \t\tsequence length.\n"
		"  -s\t\tset initial seed\n"
		"  -v\t\tdisplay version and exit\n");
}

static void print_version()
{
	printf("iimc version 0.1\n");
}

static void parse_cmd(int argc, char *argv[], struct iimc_cfg *p)
{
	if (argc < 2)
		return;

	int opt;
	while ((opt = getopt(argc, argv, "hvn:s:m:r:l:")) != -1) {
		switch (opt) {
			case 'h':
				print_help();
				exit(EXIT_SUCCESS);
			case 'l':
				p->seq_len = atoi(optarg);
				break;
			case 'm':
				p->mf = optarg;
				break;
			case 'n':
				p->num_token = atoi(optarg);
				break;
			case 'r':
				sscanf(optarg, "%3f", &p->oversize_r);
				break;
			case 's':
				p->rng_state = atoi(optarg);
				break;
			case 'v':
				print_version();
				exit(EXIT_SUCCESS);
#if 0
			case 'p':
				p->prompt = optarg;
				break;
#endif
		}
	}
}

static int check_prompt(struct token_buffer *tb, const char *prompt)
{
	assert(tb != NULL);
	assert(tb->buf != NULL);

	if (prompt == NULL)
		return 0;

#if 0
	size_t maxlen = tb->max_seq_len - 1;
	size_t len = strnlen(prompt, maxlen);
	if (len == tb->max_seq_len)
		return -1;

	strncpy(&tb->buf[1], prompt, len);
	tb->last_pos = len;
#endif
	return 0;
}

int main(int argc, char *argv[])
{
	struct iimc_cfg cfg;
	iimc_cfg_default(&cfg);

	parse_cmd(argc, argv, &cfg);

	struct iimc_gpt2 *m = iimc_gpt2_new();
	if (m == NULL) {
		fprintf(stderr, "Failed to allocate memory for model. "
				"Likely out of memory.\n");
		exit(EXIT_FAILURE);
	}

	int r;
	
	r = iimc_gpt2_load(m, cfg.mf);
	switch (r) {
		case IIMC_EFILE_NOT_FOUND:
			fprintf(stderr, "Failed to load model. "
					"File not found.\n");
			exit(EXIT_FAILURE);
		case IIMC_EFILE_BAD_HEADER:
			fprintf(stderr, "Failed to load model. "
					"Model file has a bad header.\n");
			exit(EXIT_FAILURE);
		case IIMC_ENOMEM:
			fprintf(stderr, "Failed to load model. "
					"Memory allocation error.\n");
			exit(EXIT_FAILURE);
		case IIMC_ENONE:
			break;
		default:
			fprintf(stderr, "Failed to load model. "
					"Unknown error.\n");
			exit(EXIT_FAILURE);
	}

	if (cfg.seq_len < 1)
		cfg.seq_len = m->cfg.max_seq_len;

	r = iimc_gpt2_init(m, 1, cfg.seq_len);
	switch (r) {
		case IIMC_ENOMEM:
			fprintf(stderr, "Failed to init model. "
					"Memory allocation error.\n");
			exit(EXIT_FAILURE);
		case IIMC_ENONE:
			break;
		default:
			fprintf(stderr, "Failed to init model. "
					"Unknown error.\n");
			exit(EXIT_FAILURE);
	}

	struct token_buffer *tb = token_buffer_new(cfg.seq_len, cfg.oversize_r);
	if (tb == NULL) {
		fprintf(stderr, "Failed to init token buffer.\n");
		exit(EXIT_FAILURE);
	}

	if (check_prompt(tb, cfg.prompt)  != 0) {
		fprintf(stderr, "Failed to set initial prompt. "
				"Possibly prompt is too long.");
		exit(EXIT_FAILURE);
	}

	int indx;
	for (int t = 1; t != cfg.num_token + 1; t++) {
		int *buffer = token_buffer_step(tb, &indx);
		iimc_gpt2_forward(m, buffer, NULL, 1, indx);
		int value = iimc_gpt2_sample(m, indx, &cfg.rng_state);
		token_buffer_update(tb, value);
		printf("%d ", value);
		fflush(stdout);
	}
	printf("\n");

	token_buffer_free(tb);
	iimc_gpt2_free(m);
	return 0;
}
