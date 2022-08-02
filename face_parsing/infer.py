from queue import Queue # Python 3.x
from threading import Thread

def foo(bar):
    print('hello {0}'.format(bar))   # Python 3.x
    return 'foo'

que = Queue()           # Python 3.x

threads_list = list()

t1 = Thread(target=lambda q, arg1: q.put(foo(arg1)), args=(que, 'world!'))
t1.start()
threads_list.append(t1)

# Add more threads here
t2 = Thread(target=lambda q, arg1: q.put(foo(arg1)), args=(que, 'abc!'))
t2.start()
threads_list.append(t2)

t3 = Thread(target=lambda q, arg1: q.put(foo(arg1)), args=(que, 'pqr!'))
t3.start()
threads_list.append(t3)

# Join all the threads
for t in threads_list:
    t.join()

# Check thread's return value
while not que.empty():
    result = que.get()
    print(result)       # Python 3.x