#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <math.h>

#include "iimc.h"

struct iimc_gpt2 *iimc_gpt2_new(void)
{
	struct iimc_gpt2 *m = malloc(sizeof(struct iimc_gpt2));
	if (m == NULL)
		return NULL;

	memset(m, 0, sizeof(struct iimc_gpt2));
	return m;
}

int iimc_gpt2_free(struct iimc_gpt2 *m)
{
	if (m == NULL) 
		return IIMC_ENULL_POINTER_FREE;

	if (m->params != NULL)
		free(m->params);

	if (m->acts != NULL)
		free(m->acts);

	memset(m, 0, sizeof(struct iimc_gpt2));
	free(m);
	return IIMC_ENONE;
}

static int model_load_header(struct iimc_gpt2 *m, FILE *mf)
{
	assert(m != NULL);
	assert(mf != NULL);

	int header[256];
	if (fread(header, sizeof(header), 1, mf) != 1)
		return IIMC_EFILE_BAD_HEADER;

	if (header[0] != 20240326) 
		return IIMC_EFILE_BAD_HEADER;

	if (header[1] != 1)
		return IIMC_EFILE_BAD_HEADER;

	m->cfg.max_seq_len	= header[2];
	m->cfg.vocab_size	= header[3];
	m->cfg.num_layers	= header[4];
	m->cfg.num_heads	= header[5];
	m->cfg.channels		= header[6];

	return IIMC_ENONE;
}

static size_t model_count_param(struct iimc_gpt2 *m)
{
	assert(m != NULL);

	size_t count = 0;
	int i;
	for (i = 0; i < NUM_PARAMETER_TENSORS; i++)
		count += m->param_size[i];

	return count;
}

static void model_update_param_count(struct iimc_gpt2 *m)
{
	assert(m != NULL);

	m->param_count = model_count_param(m);
	m->param_bytes = m->param_count * sizeof(float);
}

static size_t model_load_param_sizes(struct iimc_gpt2 *m)
{
	assert(m != NULL);

	size_t lc = m->cfg.num_layers * m->cfg.channels;

	m->param_size[ 0] = m->cfg.vocab_size * m->cfg.channels;
	m->param_size[ 1] = m->cfg.max_seq_len * m->cfg.channels;
	m->param_size[ 2] = lc;
	m->param_size[ 3] = lc;
	m->param_size[ 4] = lc * 3 * m->cfg.channels;
	m->param_size[ 5] = lc * 3;
	m->param_size[ 6] = lc * m->cfg.channels;
	m->param_size[ 7] = lc;
	m->param_size[ 8] = lc;
	m->param_size[ 9] = lc;
	m->param_size[10] = lc * 4 * m->cfg.channels;
	m->param_size[11] = lc * 4;
	m->param_size[12] = lc * 4 * m->cfg.channels;
	m->param_size[13] = lc;
	m->param_size[14] = m->cfg.channels;
	m->param_size[15] = m->cfg.channels;

	model_update_param_count(m);
	return m->param_count;
}

static int model_load_params_new(struct iimc_gpt2 *m)
{
	assert(m != NULL);

	/* The address of a block returned by malloc or realloc in GNU systems 
	 * is always a multiple of eight (or sixteen on 64-bit systems).
	 *
	 * But to be explicit, I use posix_memalign.
	 *
	 * see
	 * www.gnu.org/software/libc/manual/html_node/Aligned-Memory-Blocks.html
	 * man posix_memalign
	 */

	if (m->params != NULL)
		free(m->params);

	int r = posix_memalign((void **) &m->params, 64, m->param_bytes);
	switch (r) {
		case 0:
			break;
		case ENOMEM:
			return IIMC_ENOMEM;
		default:
			return IIMC_EUNKNOWN;
	}

	float *p = m->params;
	m->param.wte = p;
	p += m->param_size[ 0]; m->param.wpe = p;
	p += m->param_size[ 1]; m->param.ln1w = p;
	p += m->param_size[ 2]; m->param.ln1b = p;
	p += m->param_size[ 3]; m->param.qkvw = p;
	p += m->param_size[ 4]; m->param.qkvb = p;
	p += m->param_size[ 5]; m->param.attprojw = p;
	p += m->param_size[ 6]; m->param.attprojb = p;
	p += m->param_size[ 7]; m->param.ln2w = p;
	p += m->param_size[ 8]; m->param.ln2b = p;
	p += m->param_size[ 9]; m->param.fcw = p;
	p += m->param_size[10]; m->param.fcb = p;
	p += m->param_size[11]; m->param.fcprojw = p;
	p += m->param_size[12]; m->param.fcprojb = p;
	p += m->param_size[13]; m->param.lnfw = p;
	p += m->param_size[14]; m->param.lnfb = p;

	return IIMC_ENONE;
}

static int model_load_params(struct iimc_gpt2 *m, FILE *mf)
{
	assert(m != NULL);
	assert(mf != NULL);

	/*
	 * Make sure that the programmer understands the following.
	 * fread fails if:
	 * the buffer size is less than nmemb * size
	 * the number of unread bytes is less than nmemb * size
	 *
	 * It is important that the buffer size, number of unread bytes,
	 * and nmemb * size match.
	 */
	if (fread(m->params, m->param_bytes, 1, mf) != 1)
		return IIMC_EFILE_BAD_PARAMS;

	return IIMC_ENONE;
}

int iimc_gpt2_load(struct iimc_gpt2 *m, const char *path)
{
	assert(m != NULL);
	assert(path != NULL);

	FILE *mf = fopen(path, "rb");
	if (mf == NULL)
		return IIMC_EFILE_NOT_FOUND;

	if (model_load_header(m, mf) != IIMC_ENONE) {
		fclose(mf);
		return IIMC_EFILE_BAD_HEADER;
	}

	if (model_load_param_sizes(m) == 0) {
		fclose(mf);
		return IIMC_EFILE_BAD_HEADER;
	}

	int r;

	r = model_load_params_new(m);
	if (r != IIMC_ENONE) {
		fclose(mf);
		free(m);
		return r;
	}

	r = model_load_params(m, mf);
	if (r != IIMC_ENONE) {
		fclose(mf);
		free(m);
		return r;
	}

	fclose(mf);

	return IIMC_ENONE;
}

static int model_init_acts_new(struct iimc_gpt2 *m)
{
	assert(m != NULL);

	if (m->acts != NULL)
		free(m->acts);

	int r = posix_memalign((void **) &m->acts, 64, m->act_bytes);
	switch (r) {
		case 0:
			break;
		case ENOMEM:
			return IIMC_ENOMEM;
		default:
			return IIMC_EUNKNOWN;
	}

	float *p = m->acts;
	m->act.encoded = p;
	p += m->act_size[ 0]; m->act.ln1 = p;
	p += m->act_size[ 1]; m->act.ln1_mean = p;
	p += m->act_size[ 2]; m->act.ln1_rstd = p;
	p += m->act_size[ 3]; m->act.qkv = p;
	p += m->act_size[ 4]; m->act.atty = p;
	p += m->act_size[ 5]; m->act.preatt = p;
	p += m->act_size[ 6]; m->act.att = p;
	p += m->act_size[ 7]; m->act.attproj = p;
	p += m->act_size[ 8]; m->act.residual2 = p;
	p += m->act_size[ 9]; m->act.ln2 = p;
	p += m->act_size[10]; m->act.ln2_mean = p;
	p += m->act_size[11]; m->act.ln2_rstd = p;
	p += m->act_size[12]; m->act.fch = p;
	p += m->act_size[13]; m->act.fch_gelu = p;
	p += m->act_size[14]; m->act.fcproj = p;
	p += m->act_size[15]; m->act.residual3 = p;
	p += m->act_size[16]; m->act.lnf = p;
	p += m->act_size[17]; m->act.lnf_mean = p;
	p += m->act_size[18]; m->act.lnf_rstd = p;
	p += m->act_size[19]; m->act.logits = p;
	p += m->act_size[20]; m->act.probs = p;
	p += m->act_size[21]; m->act.losses = p;

	return IIMC_ENONE;
}

static size_t model_count_act(struct iimc_gpt2 *m)
{
	assert(m != NULL);

	size_t count = 0;
	int i;
	for (i = 0; i < NUM_ACTIVATION_TENSORS; i++)
		count += m->act_size[i];

	return count;
}

static void model_update_act_count(struct iimc_gpt2 *m)
{
	assert(m != NULL);

	m->act_count = model_count_act(m);
	m->act_bytes = m->act_count * sizeof(float);
}


static int model_init_acts(struct iimc_gpt2 *m, int b, int t)
{
	size_t bt = b * t;
	int v = m->cfg.vocab_size;
	int l = m->cfg.num_layers;
	int nh = m->cfg.num_heads;
	int c = m->cfg.channels;
	m->act_size[ 0] = bt * c;
	m->act_size[ 1] = l * bt * c;
	m->act_size[ 2] = l * bt;
	m->act_size[ 3] = l * bt;
	m->act_size[ 4] = l * bt * c * 3;
	m->act_size[ 5] = l * bt * c;
	m->act_size[ 6] = l * bt * nh * t;
	m->act_size[ 7] = l * bt * nh * t;
	m->act_size[ 8] = l * bt * c;
	m->act_size[ 9] = l * bt * c;
	m->act_size[10] = l * bt * c;
	m->act_size[11] = l * bt;
	m->act_size[12] = l * bt;
	m->act_size[13] = l * bt * c * 4;
	m->act_size[14] = l * bt * c * 4;
	m->act_size[15] = l * bt * c;
	m->act_size[16] = l * bt * c;
	m->act_size[17] = bt * c;
	m->act_size[18] = bt;
	m->act_size[19] = bt;
	m->act_size[20] = bt * v;
	m->act_size[21] = bt * v;
	m->act_size[22] = bt;

	model_update_act_count(m);

	return IIMC_ENONE;
}

int iimc_gpt2_init(struct iimc_gpt2 *m, int b, int t)
{
	assert(m != NULL);
	int r;

	r = model_init_acts(m, b, t);
	if (r != IIMC_ENONE)
		return r;

	r = model_init_acts_new(m);
	if (r != IIMC_ENONE)
		return r;

	return IIMC_ENONE;
}

static void encoder_forward(float *out, int *in, float *wte, float *wpe,
	       	int b, int t, int c)
{
	int i, j, k;

	for (i = 0; i < b; i++) {
		for (j = 0; j < t; j++) {
			float *o = out + i * t * c + j * c;
			int ix = in[i * t + j];
			float *wte_ix = wte + ix * c;
			float *wpe_t = wpe + j * c;
			for (k = 0; k < c; k++) {
				o[k] = wte_ix[k] + wpe_t[k];
			}
		}
	}
}

static void layernorm_forward(float *out, float *mean, float *rstd, float *inp,
		float *weight, float *bias, int b, int t, int c)
{
	float eps = 1e-5f;
	int i, j, k;

	for (i = 0; i < b; i++) {
		for (j = 0; j < t; j++) {
			float *x = inp + i * t * c + j * c;
			float m = 0.0f;
			for (k = 0; k < c; k++) {
				m += x[k];
			}
			m = m / c;

			float v = 0.0f;
			for (k = 0; k < c; k++) {
				float xshift = x[k] - m;
				v += xshift * xshift;
			}
			v = v / c;

			float s = 1.0f / sqrtf(v + eps);

			float *o = out + i * t * c + j * c;
			for (k = 0; k < c; k++) {
				float n = s * (x[k] - m);
				o[k] = n * weight[k] + bias[k];
			}
			mean[i * t + j] = m;
			rstd[i * t + j] = s;
		}
	}
}

static void matmul_forward(float *out, float *inp, float *weight, float *bias,
		int b, int t, int c, int oc)
{
	int i, j, k, m;
#pragma omp parallel for collapse(2)
	for (i = 0; i < b; i++) {
		for (j = 0; j < t; j++) {
			float *out_bt = out + i * t * oc + j * oc;
			float *inp_bt = inp + i * t * c + j * c;
			for (k = 0; k < oc; k++) {
				/* val = (bias != NULL) ? bias[k] : 0.0f; */
				float val = bias[k];
				float *wrow = weight + k * c;
				for (m = 0; m < c; m++) {
					val += inp_bt[m] * wrow[m];
				}
				out_bt[k] = val;
			}
		}
	}
}

static void matmul_forward_nobias(float *out, float *inp, float *weight, 
		int b, int t, int c, int oc)
{
	int i, j, k, m;
#pragma omp parallel for collapse(2)
	for (i = 0; i < b; i++) {
		for (j = 0; j < t; j++) {
			float *out_bt = out + i * t * oc + j * oc;
			float *inp_bt = inp + i * t * c + j * c;
			for (k = 0; k < oc; k++) {
				float val = 0.0f;
				float *wrow = weight + k * c;
				for (m = 0; m < c; m++) {
					val += inp_bt[m] * wrow[m];
				}
				out_bt[k] = val;
			}
		}
	}
}


void attention_forward(float *out, float *preatt, float *att, float *inp,
		int b, int t, int c, int nh)
{
	int c3 = 3 * c;
	int hs = c / nh;
	float scale = 1.0f / sqrtf(hs); 

	int i, j, k, m, n;

#pragma omp parallel for collapse(3)
	for (i = 0; i < b; i++) {
	for (j = 0; j < t; j++) {
	for (k = 0; k < nh; k++) {
		float *query_t = inp + i * t * c3 + j * c3 + k * hs;
		float *preatt_bth = preatt + i * nh * t * t + k * t * t + j * t;
		float *att_bth = att + i * nh * t * t + k * t * t + j * t;

		/* pass 1 */
		float maxval = -10000.0f;
		for (m = 0; m <= j; m++) {
			float *key_t2 = inp + i * t * c3 + 
				m * c3 + k * hs + c;
			float val = 0.0f;
			for (n = 0; n < hs; n++) {
				val += query_t[n] * key_t2[n];
			}
			val *= scale;
			if (val > maxval) maxval = val;
			preatt_bth[m] = val;
		}

		/* pass 2 */
		float expsum = 0.0f;
		for (m = 0; m <= j; m++) {
			float expv = expf(preatt_bth[m] - maxval);
			expsum += expv;
			att_bth[m] = expv;
		}

		float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

		/* pass 3 */
		for (m = 0; m < t; m++) {
			if (m <= j) {
				att_bth[m] *= expsum_inv;
			} else {
				att_bth[m] = 0.0f;
			}
		}

		/* pass 4 */
		float *out_bth = out + i * t * c + j * c + k * hs;
		for (m = 0; m < hs; m++) {
			out_bth[m] = 0.0f;
		}
		for (m = 0; m <= j; m++) {
			float *value_t2 = inp + i * t * c3 + m * c3
				+ k * hs + 2 * c;
			float att_btht2 = att_bth[m];
			for (n = 0; n < hs; n++) {
				out_bth[n] += att_btht2 * value_t2[n];
			}
		}
	}
	}
	}
}

void gelu_forward(float *out, float *inp, int n)
{
	const float s = sqrt(2.0f / M_PI);
	int i;
	for (i = 0; i < n; i++) {
		float x = inp[i];
		float cube = 0.044715f * x * x * x;
		out[i] = 0.5f * x * (1.0f + tanhf(s * (x + cube)));
	}
}

void residual_forward(float *out, float *inp1, float *inp2, int n)
{
	int i;
	for (i = 0; i < n; i++)
		out[i] = inp1[i] + inp2[i];
}

void softmax_forward(float *probs, float *logits, int b, int t, int v)
{
	int i, j, k;
#pragma omp parallel for collapse(2)
	for (i = 0; i < b; ++i) {
		for (j = 0; j < t; j++) {
			float *logits_bt = logits + i * t * v + j * v;
			float *probs_bt = probs + i * t * v + j * v;
			float maxval = -10000.0f;
			for (k = 0; k < v; k++) {
				if (logits_bt[k] > maxval) {
					maxval = logits_bt[k];
				}
			}
			float sum = 0.0f;
			for (k = 0; k < v; k++) {
				probs_bt[k] = expf(logits_bt[k] - maxval);
				sum += probs_bt[k];
			}
			for (k = 0; k < v; k++) {
				probs_bt[k] /= sum;
			}
		}
	}
}

int iimc_gpt2_forward(struct iimc_gpt2 *m, int *in, int *target, int b, int t)
{
	assert(m != NULL);
	assert(in != NULL);

	int bt = b * t;
	int btc = bt * m->cfg.channels;

	encoder_forward(m->act.encoded, in, m->param.wte, m->param.wpe,
			b, t, m->cfg.channels);

#if 1
	/* unrolled i = 0 */
	layernorm_forward(m->act.ln1, m->act.ln1_mean, m->act.ln1_rstd,
			m->act.encoded, m->param.ln1w, m->param.ln1b,
			b, t, m->cfg.channels);
	matmul_forward(m->act.qkv, m->act.ln1, m->param.qkvw, m->param.qkvb,
			b, t, m->cfg.channels, 3 * m->cfg.channels);
	attention_forward(m->act.atty, m->act.preatt, m->act.att, m->act.qkv,
			b, t, m->cfg.channels, m->cfg.num_heads);
	matmul_forward(m->act.attproj, m->act.atty, m->param.attprojw,
			m->param.attprojb, b, t,
			m->cfg.channels, m->cfg.channels);
	residual_forward(m->act.residual2, m->act.encoded,
			m->act.attproj, btc);
	layernorm_forward(m->act.ln2, m->act.ln2_mean, m->act.ln2_rstd,
			m->act.residual2, m->param.ln2w, m->param.ln2b,
			b, t, m->cfg.channels);
	matmul_forward(m->act.fch, m->act.ln2, m->param.fcw, m->param.fcb,
			b, t, m->cfg.channels, 4 * m->cfg.channels);
	gelu_forward(m->act.fch_gelu, m->act.fch, 4 * btc);
	matmul_forward(m->act.fcproj, m->act.fch_gelu,
			m->param.fcprojw, m->param.fcprojb,
			b, t, 4 * m->cfg.channels, m->cfg.channels);
	residual_forward(m->act.residual3, m->act.residual2,
			m->act.fcproj, btc);
#endif

	int i;
	for (i = 1; i < m->cfg.num_layers; i++) {
		float *residual = m->act.residual3 + (i - 1) * btc;

		int ibtc = i * btc;
		int ibt = i * bt;
		int ic = i * m->cfg.channels;

		float *l_ln1 = m->act.ln1 + ibtc;
		float *l_qkv = m->act.qkv + ibtc * 3;
		float *l_atty = m->act.atty + ibtc;
		float *l_attproj = m->act.attproj + ibtc;
		float *l_residual2 = m->act.residual2 + ibtc;
		float *l_ln2 = m->act.ln2 + ibtc;
		float *l_fch = m->act.fch + ibtc * 4;
		float *l_fch_gelu = m->act.fch_gelu + ibtc * 4;
		float *l_fcproj = m->act.fcproj + ibtc;

		layernorm_forward(l_ln1, m->act.ln1_mean + ibt,
				m->act.ln1_rstd + ibt, residual, 
				m->param.ln1w + ic, 
				m->param.ln1b + ic,
				b, t, m->cfg.channels);
		matmul_forward(l_qkv, l_ln1,
				m->param.qkvw + ic * 3 * m->cfg.channels, 
				m->param.qkvb + ic * 3, b, t, 
				m->cfg.channels, m->cfg.channels * 3);
		attention_forward(l_atty,
				m->act.preatt + ibt * t * m->cfg.num_heads,
				m->act.att + ibt * t * m->cfg.num_heads, 
				l_qkv, b, t, m->cfg.channels,
				m->cfg.num_heads);
		matmul_forward(m->act.attproj + ibtc, l_atty, 
				m->param.attprojw + ic * m->cfg.channels,
				m->param.attprojb + ic,
				b, t, m->cfg.channels, m->cfg.channels);
		residual_forward(l_residual2, residual, l_attproj, btc);
		layernorm_forward(l_ln2, m->act.ln2_mean + ibt, 
				m->act.ln2_rstd + ibt, l_residual2, 
				m->param.ln2w + ic, 
				m->param.ln2b + ic,
				b, t, m->cfg.channels);
		matmul_forward(l_fch, l_ln2,
				m->param.fcw + ic * 4 * m->cfg.channels,
				m->param.fcb + ic * 4, 
				b, t, m->cfg.channels, 4 * m->cfg.channels);
		gelu_forward(l_fch_gelu, l_fch, btc * 4);
		matmul_forward(l_fcproj, l_fch_gelu, 
				m->param.fcprojw + ic * m->cfg.channels * 4,
				m->param.fcprojb + ic,
				b, t, 4 * m->cfg.channels, m->cfg.channels);
		residual_forward(m->act.residual3 + ibtc, l_residual2,
				l_fcproj, btc);
	}

	float *residual = m->act.residual3 + (m->cfg.num_layers - 1) * btc;
	layernorm_forward(m->act.lnf, m->act.lnf_mean,
			m->act.lnf_rstd, residual,
			m->param.lnfw, m->param.lnfb,
			b, t, m->cfg.channels);
	matmul_forward_nobias(m->act.logits, m->act.lnf, m->param.wte, 
			b, t, m->cfg.channels, m->cfg.vocab_size);
	softmax_forward(m->act.probs, m->act.logits, b, t, m->cfg.vocab_size);

	return IIMC_ENONE;
}

static inline unsigned int random_u32(unsigned long long *state)
{
	*state ^= *state >> 12;
	*state ^= *state << 25;
	*state ^= *state >> 27;
	return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

static float random_f32(unsigned long long *state)
{
	return (random_u32(state) >> 8) / 16777216.0f;
}

static int sample_mult(float *prob, int n, float coin)
{
	float cdf = 0.0f;
	int i;
	for (i = 0; i < n; i++) {
		cdf += prob[i];
		if (coin < cdf) {
			return i;
		}
	}
	return n - 1;
}

extern int iimc_gpt2_sample(struct iimc_gpt2 *m, int t,
		unsigned long long *rng_state)
{
	float *probs = m->act.probs + (t - 1) * m->cfg.vocab_size;
	float coin = random_f32(rng_state);
	return sample_mult(probs, m->cfg.vocab_size, coin);
}
