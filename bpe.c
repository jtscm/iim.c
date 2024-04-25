#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>

#include "iimc.h"

struct iimc_bpe {
	unsigned int vocab_count;
	size_t max_word_size;
	size_t dec_size;
	char *dec;
};

struct iimc_bpe *iimc_bpe_new(void)
{
	struct iimc_bpe *t = malloc(sizeof(struct iimc_bpe));
	if (t == NULL)
		return NULL;

	memset(t, 0, sizeof(struct iimc_bpe));
	return t;
}

int iimc_bpe_free(struct iimc_bpe *p)
{
	if (p == NULL)
		return IIMC_ENULL_POINTER_FREE;

	if (p->dec != NULL)
		free(p->dec);

	memset(p, 0, sizeof(struct iimc_bpe));
	return IIMC_ENONE;
}

static int bpe_load_header(struct iimc_bpe *t, FILE *stream)
{
	unsigned int header[256];
	if (fread(header, sizeof(header), 1, stream) != 1)
		return IIMC_EFILE_BAD_HEADER;

	if (header[0] != 20240328) 
		return IIMC_EFILE_BAD_HEADER;

	if (header[1] != 1)
		return IIMC_EFILE_BAD_HEADER;

	t->vocab_count = header[2]; /* 50257 */
	if (t->vocab_count == 0)
		return IIMC_EFILE_BAD_HEADER;

	/* +1 is used for terminating null character */
	t->max_word_size = 0x80 + 0x01; 

	/* for GPT2 the vocab size is around 6.1 MB */
	t->dec_size = t->max_word_size * t->vocab_count;

	return IIMC_ENONE;
}

static int bpe_load_data(struct iimc_bpe *t, FILE *stream)
{
	unsigned int i; 

	/* pass for decoding */
	for (i = 0; i < t->vocab_count; i++) {
		int size = 0;
		int c = fread(&size, 1, 1, stream);
		if (c != 1)
			return IIMC_EFILE_UNEXPECTED_EOF;

		if (size > (t->max_word_size - 1))
			return IIMC_EFILE_BAD_WORD_SIZE;

		if (fread(&t->dec[i * t->max_word_size], 1, size, stream) != size)
			return IIMC_EFILE_BAD_TOKENS;
	}

	return IIMC_ENONE;
}

int iimc_bpe_load(struct iimc_bpe *t, const char *model)
{
	FILE *stream = fopen(model, "r");
	if (stream == NULL)
		return IIMC_EFILE_NOT_FOUND;

	bpe_load_header(t, stream);

	t->dec = malloc(t->dec_size);
	if (t->dec == NULL) {
		fclose(stream);
		return IIMC_ENOMEM;
	}

	int r;
	r = bpe_load_data(t, stream);
	fclose(stream);

	if (r != IIMC_ENONE) {
		free(t->dec);
		return r;
	}

	return IIMC_ENONE;
}

char *iimc_bpe_decode(struct iimc_bpe *t, int value)
{
	assert(t != NULL);
	if (value < 0 || value > t->dec_size)
		return NULL;

	return &t->dec[value * t->max_word_size];
}
