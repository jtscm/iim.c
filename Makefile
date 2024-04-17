CC = gcc
CFLAGS = -O3 -g -Wall -march=native
LDFLAGS = -mavx -mavx2 -mfma
LDLIBS = -lm 
INCLUDES =
TARGET = iimc
SRC = iimc.c main.c
OBJ = $(SRC:.c=.o)

CFLAGS += -fopenmp -DOMP
LDLIBS += -lgomp

$(TARGET): $(OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) $(LDFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) $(TARGET)
