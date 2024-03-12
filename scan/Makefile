CC = nvcc
CFLAGS =
LDFLAGS = 

TARGET = scan.out

OBJS = scan.o

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