#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/sem.h>

#define N 5
#define THINKING 0
#define HUNGRY 1
#define EATING 2

#define LEFT (i + N-1) % N
#define RIGHT (i + 1) % N

int sem_set_id;
int i;

struct shm{
    int state[N];
}*shared_memory;

void sharedMemory(){
    int shmid;
	shmid = shmget(IPC_PRIVATE, sizeof(*shared_memory), 0644|IPC_CREAT);
    shared_memory = (struct shm*)shmat(shmid, NULL, 0);
    if (shmid == -1) {
      printf("\nError");
      return 1;
    }else{
        printf("\nShmat succeed");
    }
}

void up(int sem_no) {
    struct sembuf semops;
    semops.sem_num = sem_no;
    semops.sem_op = 1;
    semops.sem_flg = SEM_UNDO;
    semop(sem_set_id, &semops, 1);
}

void down(int sem_no) {
    struct sembuf semops;
    semops.sem_num = sem_no;
    semops.sem_op = -1;
    semops.sem_flg = SEM_UNDO;
    semop(sem_set_id, &semops, 1);
}

void test(int i){
	if (shared_memory->state[i] == HUNGRY && shared_memory->state[LEFT] != EATING && shared_memory->state[RIGHT] != EATING){
		shared_memory->state[i]=EATING;
		up(i);
	}
}

void take_forks(int i){
	down(N);
	printf("Philosopher [%d] is hungry.\n", i);
	shared_memory->state[i]=HUNGRY;
	test(i);
	up(N);
	down(i);
}

void put_forks(int i){
	down(N);
	shared_memory->state[i]=THINKING;
	test(LEFT);
	test(RIGHT);
	up(N);
}

void think(){
	printf("Philosopher [%d] is thinking.\n", i);
}

void eat(){
	printf("Philosopher [%d] is eating.\n", i);
}

void philosopher(int i){
	while(1){
		think();
		sleep(1);
		take_forks(i);
		eat();
		sleep(1);
		put_forks(i);
	}
}

int main(){
	int j;
	//Initialize Shared memory
	sharedMemory();

	//Initialize Semaphores
	sem_set_id = semget(IPC_PRIVATE, N + 1, 0644|IPC_CREAT);
	printf("\nSemaphores group id: %d\n", sem_set_id);
	if (sem_set_id == -1) {
		printf("\nError:semaphore");
		exit(1);
	}

	ushort semvals[N];
    union semun  {
        int val;
        struct semid_ds *buf;
        ushort *array;
    } arg;
    semctl(sem_set_id, 0, IPC_STAT, arg);
	for(j = 0; j < N ; j++)semvals[j] = 0;
	arg.array = semvals;
    semctl(sem_set_id, 0, SETALL, arg);
    arg.val = 1;
    semctl(sem_set_id, j, SETVAL, arg);

	int pid;
	for(i = 0; i < N; i++){
		pid=fork();
		if (pid == 0) break;
	}

	if (pid == 0){
		philosopher(i);
	} else if (pid < 0){
		printf("Error: Cannot create child processes\n");
		exit(1);
	} else {
		wait(NULL);
	}


	return 0;
}