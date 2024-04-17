# iim.c
Inference for llm.c

Recently, Andrej Karpathy released llm.c, a C program for training GPT-2.

Here, I edited llm.c (CPU) for inference.

What is different:
- linux coding style;
- only the forward pass;
- iim.c accepts command line arguments;

To run iim.c, acquire the model .bin file by Karpathy's llm.c.
Please note you may need to edit the Makefile to compile.
 
TODO:
- token decoding;
- token encoding;
- writing to the input buffer by stdin;

Date: April, 17th 2024.
