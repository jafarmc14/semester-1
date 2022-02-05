#include<stdio.h>
#include<sys/wait.h>
#include<unistd.h>
#include<stdlib.h>
#include<signal.h>

#define NUM_CHILD 3

#define with_signal

#ifdef with_signal
int mark = 0;

void ignore_handler(int signal_){
    printf("\nProcess[%d] cannot be interrupted", getpid());
}

void handler_parent(int signal_){
	printf("\nchild[%d] from parent[%d] is interrupted", getpid(), getppid());
    mark++;
}

void handler_child(int signal_){
    printf("\nchild[%d] from parent[%d] is interrupted", getpid(), getppid());
}

#endif

int childAlgorithm(){
	pid_t processId = getpid();
	pid_t parentId = getppid();
	printf("\nchild[%d] from parent[%d] started", processId, parentId);	
	sleep(10);
	printf("\nchild[%d] from parent[%d] finished", processId, parentId);
	exit(0);
}


void main(){
    
    int num = 0, j, id_store[NUM_CHILD];

#ifdef with_signal
    int x;
#endif

	for(int i = 0; i < NUM_CHILD; i++){
	    pid_t id1 = fork();
		//one second delays between consecutive fork()
		sleep(1);
	    
#ifdef with_signal
        // check the mark of interrupt
        if(mark)
        {
            for (j = 0; j < i - 1; j++)
            {
                kill(id_store[j], SIGTERM);
                printf("\nParent[%d] sends SIGTERM Signal", id_store[j]);
            }
        }
#endif	    
        fflush(stdout);

        //parent process
		if(id1 > 0){
		    printf("\nparent[%d] is processing", getpid());
#ifdef with_signal
            //ignore all of signal interrupt
            for (x = 1; x <= NSIG; x++)
            {
                signal(x, ignore_handler);
            }
            
            //restore SIGCHLD signal to default
            signal(SIGCHLD, SIG_DFL);
            
            //set the own keyboard interrupt for SIGTERM
            signal(SIGINT, handler_parent);
#endif
		}
       	//child process
		else if(id1 == 0){
		    id_store[i] = getpid();
			childAlgorithm();
#ifdef with_signal
        //ignore handler
        signal(SIGINT, ignore_handler);

        //set the own keyboard interrupt for SIGTERM
        signal(SIGTERM, handler_child);
#endif
		}
		
		//error section
		else if(id1 < 0){
			printf("\ncreation of a child was unsuccessful");
			//kill(), function sends a signal to a process
			//sigterm, termination request, sent to the program
			for(j = 0; j < i-1; j++){
			    kill(id_store[j], SIGTERM);
			    printf("\nParent[%d] send to all already created child processes SIGTERM signal", id_store[j]);
			}
           	exit(1);//Reports the abnormal termination of the program.	
		}

	}
	
	printf("\nProcess[%d] all child process are created", getpid());
	
	//Finishing the child process
	while(wait(NULL) > 0){
	    num++;//count child processes terminations
	}
	printf("\nThere are no more child processes and %d child processes finished", num);
	
	printf("\ninterrupt of the creation process: %d", mark);
	
#ifdef with_signal
    //Restore all of process to default.
    for (x = 1; x <= NSIG; x++)
    {
        signal(x, SIG_DFL);
    }
#endif
	
	exit(0);
}
