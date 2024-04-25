# iim.c
Inference for llm.c

Recently, Andrej Karpathy released llm.c, a C program for training GPT-2.

Here, I edited llm.c (CPU) for inference.

What is different:
- linux coding style;
- only the forward pass;
- iim.c accepts command line arguments;

To compile and run iim.c:
- change the Makefile to fit your system
- acquire the model gpt2_124M.bin file from llm.c
- acquire the model gpt2_tokenizer.bin file from llm.c (optional)
 
TODO:
- <s> token decoding; </s>
- token encoding;
- writing to the input buffer by stdin;
- writing to the input buffer by IPC;

Date: April, 25th 2024.
