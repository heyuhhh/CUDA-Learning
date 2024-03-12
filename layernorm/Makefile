CC = nvcc
CFLAGS = -g -G
LDFLAGS = 

TARGET = layernorm.out

OBJS = layernorm.o

all: $(TARGET)

%.o: %.cu
	$(CC) $(CFLAGS) $(LDFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)

clean:
	rm -r $(TARGET) $(OBJS)

run:
	./$(TARGET)

.PHONY: all clean