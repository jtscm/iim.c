#ifndef _IIMC_H_
#define _IIMC_H_

enum iimc_error {
	IIMC_ENONE = 0,
	IIMC_EFILE_NOT_FOUND,
	IIMC_EFILE_BAD_HEADER,
	IIMC_EFILE_BAD_PARAMS,
	IIMC_EFILE_UNEXPECTED_EOF,
	IIMC_EFILE_BAD_TOKENS,
	IIMC_EFILE_BAD_WORD_SIZE,
	IIMC_EFILE_BAD_VOCAB_BPE,
	IIMC_ENULL_POINTER_FREE,
	IIMC_ENOMEM,
	IIMC_EUNKNOWN
};

#define GPT2_EOT 50256

struct iimc_gpt2;
extern struct iimc_gpt2 *iimc_gpt2_new(void);
extern int iimc_gpt2_free(struct iimc_gpt2 *m);

extern int iimc_gpt2_load(struct iimc_gpt2 *m, const char *path);
extern int iimc_gpt2_init(struct iimc_gpt2 *m, int b, int t);
extern int iimc_gpt2_forward(struct iimc_gpt2 *m, int *in,
		int *target, int b, int t);
extern int iimc_gpt2_sample(struct iimc_gpt2 *m, int t,
		unsigned long long *rng_state);

#define NUM_PARAMETER_TENSORS	16
#define NUM_ACTIVATION_TENSORS	23
struct iimc_gpt2 {
	struct {
		int max_seq_len, vocab_size, num_layers, num_heads, channels;
	} cfg;

	size_t param_size[NUM_PARAMETER_TENSORS];
	size_t param_count;
	size_t param_bytes;
	float *params;
	struct {
		float *wte, *wpe, *ln1w, *ln1b, *qkvw, *qkvb,
		      *attprojw, *attprojb, *ln2w, *ln2b,
		      *fcw, *fcb, *fcprojw, *fcprojb, *lnfw, *lnfb;
	} param;

	size_t act_size[NUM_ACTIVATION_TENSORS];
	size_t act_count;
	size_t act_bytes;
	float *acts;
	struct {
		float *encoded, *ln1, *ln1_mean, *ln1_rstd, *qkv, *atty,
		      *preatt, *att, *attproj, *residual2, *ln2, *ln2_mean,
		      *ln2_rstd, *fch, *fch_gelu, *fcproj, *residual3,
		      *lnf, *lnf_mean, *lnf_rstd, *logits, *probs, *losses;
	} act;
};

extern struct iimc_bpe *iimc_bpe_new(void);
extern int iimc_bpe_free(struct iimc_bpe *p);
extern int iimc_bpe_load(struct iimc_bpe *p, const char *filename);
extern char *iimc_bpe_decode(struct iimc_bpe *p, int value);

#endif
