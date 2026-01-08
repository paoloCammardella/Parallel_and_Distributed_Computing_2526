#include <stdio.h>
#include <omp.h>

int main(){
    int i, id, NT;

    #pragma omp parallel for
    for(i = 0; i < 10000; ++i){
        printf("Thread %d sta aggiungendo %d\n", omp_get_thread_num(), i);

        
    }
    printf("La somma finale = %d\n", somma);
    return 0;
}